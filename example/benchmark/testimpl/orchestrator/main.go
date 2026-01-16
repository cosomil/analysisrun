package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

const (
	// 数値解析のPythonスクリプトのパスを指定する環境変数
	envScript = "SCRIPT_PATH"
	// 画像解析結果ファイルのパスを指定する環境変数
	envImageAnalysisResult = "IMAGE_ANALYSIS_RESULT"
	// 解析対象となるデータ名とサンプル名の対応表のパスを指定する環境変数
	envSamples = "SAMPLES"
	// 解析パラメータ(JSONリテラル)を指定する環境変数
	envParameters = "PARAMETERS"
	// 解析結果ファイルを出力するディレクトリを指定する環境変数
	envOutputDir = "OUTPUT_DIR"
	// 実行モードを指定する環境変数
	// "whole": 各レーンを解析する際、それぞれのプロセスに与えられた画像解析結果ファイル全体を渡す
	// "only": 各レーンを解析する際、画像解析結果ファイルから解析対象のレーンのデータのみを抽出して渡す
	envRunMode = "RUN_MODE"
)

func main() {
	// 1. モードを確認する
	runMode := strings.TrimSpace(os.Getenv(envRunMode))
	if runMode == "" {
		exitWithError(fmt.Errorf("%sが指定されていません", envRunMode))
	}
	if runMode != "whole" && runMode != "only" {
		exitWithError(fmt.Errorf("%sの値が不正です: %s", envRunMode, runMode))
	}

	// 2. 画像解析結果ファイル、対応表、パラメータを読み込む
	scriptPath := strings.TrimSpace(os.Getenv(envScript))
	if scriptPath == "" {
		exitWithError(fmt.Errorf("%sが指定されていません", envScript))
	}
	imageAnalysisPath := strings.TrimSpace(os.Getenv(envImageAnalysisResult))
	if imageAnalysisPath == "" {
		exitWithError(fmt.Errorf("%sが指定されていません", envImageAnalysisResult))
	}
	samplesPath := strings.TrimSpace(os.Getenv(envSamples))
	if samplesPath == "" {
		exitWithError(fmt.Errorf("%sが指定されていません", envSamples))
	}
	params := strings.TrimSpace(os.Getenv(envParameters))
	if params == "" {
		params = "{}"
	}
	outputDir := strings.TrimSpace(os.Getenv(envOutputDir))
	if outputDir == "" {
		exitWithError(fmt.Errorf("%sが指定されていません", envOutputDir))
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		exitWithError(err)
	}

	samplePairs, err := readSamplePairs(samplesPath)
	if err != nil {
		exitWithError(err)
	}
	if len(samplePairs) == 0 {
		exitWithError(errors.New("サンプル名CSVが空です"))
	}
	imageAnalysisBytes, err := os.ReadFile(imageAnalysisPath)
	if err != nil {
		exitWithError(err)
	}

	// 3. モードに応じた、レーンごとの画像解析結果データのファクトリを作成する
	imageAnalysisByLane := map[string][]byte{}
	if runMode == "only" {
		byLane, err := splitImageAnalysisByLane(imageAnalysisBytes)
		if err != nil {
			exitWithError(err)
		}
		for _, pair := range samplePairs {
			imageAnalysisByLane[pair.dataName] = byLane[pair.dataName]
		}
	} else {
		for _, pair := range samplePairs {
			imageAnalysisByLane[pair.dataName] = imageAnalysisBytes
		}
	}

	// 4. 3で作成したファクトリを利用して、各レーンを並列で解析(analyze)する
	results := make([]laneResult, 0, len(samplePairs))
	var resultsMu sync.Mutex
	var wg sync.WaitGroup
	errCh := make(chan error, len(samplePairs))

	for _, pair := range samplePairs {
		wg.Add(1)
		go func(pair samplePair) {
			defer wg.Done()
			laneBytes := imageAnalysisByLane[pair.dataName]
			if len(laneBytes) == 0 {
				laneBytes = imageAnalysisBytes
			}
			laneResult, err := runAnalyze(scriptPath, params, pair, laneBytes)
			if err != nil {
				errCh <- err
				return
			}
			resultsMu.Lock()
			results = append(results, laneResult)
			resultsMu.Unlock()
		}(pair)
	}

	wg.Wait()
	close(errCh)
	if err := <-errCh; err != nil {
		exitWithError(err)
	}

	// 5. 各レーンで出力された画像を指定されたディレクトリに保存する
	for _, result := range results {
		for name, data := range result.imageOutputs {
			path := filepath.Join(outputDir, name)
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				exitWithError(err)
			}
			if err := os.WriteFile(path, data, 0o644); err != nil {
				exitWithError(err)
			}
		}
	}

	// 6. 全レーンの解析が完了したら、後処理(postprocess)を実行する
	// 後処理の入力は、各レーンの解析結果を対応表に記載された順に並べてマージして作成する。
	resultByLane := map[string][]byte{}
	for _, result := range results {
		resultByLane[result.dataName] = result.analysisCSV
	}
	postprocessCSV, err := runPostprocess(scriptPath, params, resultByLane)
	if err != nil {
		exitWithError(err)
	}

	// 7. 最終的な数値解析結果を指定されたディレクトリに保存する
	resultPath := filepath.Join(outputDir, "result.csv")
	if err := os.WriteFile(resultPath, postprocessCSV, 0o644); err != nil {
		exitWithError(err)
	}
}

// scriptPath: 数値解析のPythonスクリプトのパス
// mode: 実行モード ("analyze" または "postprocess")
// out: スクリプトの標準出力先
// in: スクリプトの標準入力に渡すデータ
func run(scriptPath, mode string, out io.Writer, in io.Reader) error {
	cmd := exec.Command("uv", "run", scriptPath)
	cmd.Env = append(
		os.Environ(),
		"ANALYSISRUN_MODE="+mode,
	)
	cmd.Stdin = in
	cmd.Stdout = out
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

const imageAnalysisKey = "activity_spots"

type samplePair struct {
	dataName   string
	sampleName string
}

type laneResult struct {
	dataName     string
	sampleName   string
	analysisCSV  []byte
	imageOutputs map[string][]byte
}

type tarEntry struct {
	data   []byte
	isFile bool
}

func runAnalyze(scriptPath, params string, pair samplePair, laneCSV []byte) (laneResult, error) {
	payload := map[string]tarEntry{
		"data_name":   {data: []byte(pair.dataName)},
		"sample_name": {data: []byte(pair.sampleName)},
		"params":      {data: []byte(params)},
		"image_analysis_results/" + imageAnalysisKey: {data: laneCSV, isFile: true},
	}
	tarBytes, err := createTar(payload)
	if err != nil {
		return laneResult{}, err
	}

	var out bytes.Buffer
	if err := run(scriptPath, "analyze", &out, bytes.NewReader(tarBytes)); err != nil {
		return laneResult{}, enrichTarError(out.Bytes(), err)
	}

	entries, err := readTar(out.Bytes())
	if err != nil {
		return laneResult{}, err
	}
	if errMsg, ok := entries["error"]; ok {
		return laneResult{}, fmt.Errorf(string(errMsg.data))
	}

	analysisEntry, ok := entries["analysis_result"]
	if !ok {
		return laneResult{}, errors.New("解析結果が見つかりません")
	}

	images := map[string][]byte{}
	for name, entry := range entries {
		if after, ok := strings.CutPrefix(name, "images/"); ok {
			images[after] = entry.data
		}
	}

	return laneResult{
		dataName:     pair.dataName,
		sampleName:   pair.sampleName,
		analysisCSV:  analysisEntry.data,
		imageOutputs: images,
	}, nil
}

func runPostprocess(scriptPath, params string, results map[string][]byte) ([]byte, error) {
	payload := map[string]tarEntry{
		"params": {data: []byte(params)},
	}
	for dataName, csvBytes := range results {
		payload["analysis_results/"+dataName] = tarEntry{data: csvBytes, isFile: true}
	}

	tarBytes, err := createTar(payload)
	if err != nil {
		return nil, err
	}

	var out bytes.Buffer
	if err := run(scriptPath, "postprocess", &out, bytes.NewReader(tarBytes)); err != nil {
		return nil, enrichTarError(out.Bytes(), err)
	}
	entries, err := readTar(out.Bytes())
	if err != nil {
		return nil, err
	}
	if errMsg, ok := entries["error"]; ok {
		return nil, fmt.Errorf(string(errMsg.data))
	}
	resultEntry, ok := entries["result_csv"]
	if !ok {
		return nil, errors.New("後処理結果が見つかりません")
	}
	return resultEntry.data, nil
}

func readSamplePairs(path string) ([]samplePair, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = -1
	header, err := reader.Read()
	if err != nil {
		return nil, err
	}
	dataIdx, sampleIdx := -1, -1
	for i, name := range header {
		switch strings.TrimSpace(name) {
		case "data":
			dataIdx = i
		case "sample":
			sampleIdx = i
		}
	}
	if dataIdx < 0 || sampleIdx < 0 {
		return nil, errors.New("サンプル名CSVにdata/sample列がありません")
	}

	var pairs []samplePair
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if dataIdx >= len(record) || sampleIdx >= len(record) {
			continue
		}
		data := strings.TrimSpace(record[dataIdx])
		sample := strings.TrimSpace(record[sampleIdx])
		if data == "" || sample == "" {
			continue
		}
		pairs = append(pairs, samplePair{dataName: data, sampleName: sample})
	}
	return pairs, nil
}

func splitImageAnalysisByLane(csvBytes []byte) (map[string][]byte, error) {
	reader := csv.NewReader(bytes.NewReader(csvBytes))
	reader.FieldsPerRecord = -1
	header, err := reader.Read()
	if err != nil {
		return nil, err
	}
	filenameIdx := -1
	for i, name := range header {
		if strings.TrimSpace(name) == "Filename" {
			filenameIdx = i
			break
		}
	}
	if filenameIdx < 0 {
		return nil, errors.New("画像解析結果CSVにFilename列がありません")
	}

	grouped := map[string][][]string{}
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if filenameIdx >= len(record) {
			continue
		}
		dataName := extractLaneName(record[filenameIdx])
		grouped[dataName] = append(grouped[dataName], record)
	}

	result := map[string][]byte{}
	for dataName, rows := range grouped {
		buf, err := buildCSV(header, rows)
		if err != nil {
			return nil, err
		}
		result[dataName] = buf
	}
	return result, nil
}

func buildCSV(header []string, rows [][]string) ([]byte, error) {
	var buf bytes.Buffer
	writer := csv.NewWriter(&buf)
	if err := writer.Write(header); err != nil {
		return nil, err
	}
	for _, row := range rows {
		if err := writer.Write(row); err != nil {
			return nil, err
		}
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func extractLaneName(filename string) string {
	parts := strings.SplitN(filename, "_000_", 2)
	if len(parts) != 2 {
		return ""
	}
	rest := parts[1]
	if idx := strings.Index(rest, "."); idx >= 0 {
		rest = rest[:idx]
	}
	return rest
}

func createTar(entries map[string]tarEntry) ([]byte, error) {
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gw)

	names := make([]string, 0, len(entries))
	for name := range entries {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		entry := entries[name]
		hdr := &tar.Header{
			Name: name,
			Mode: 0o600,
			Size: int64(len(entry.data)),
		}
		if entry.isFile {
			hdr.PAXRecords = map[string]string{"is_file": "true"}
		}
		if err := tw.WriteHeader(hdr); err != nil {
			return nil, err
		}
		if _, err := tw.Write(entry.data); err != nil {
			return nil, err
		}
	}
	if err := tw.Close(); err != nil {
		return nil, err
	}
	if err := gw.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func readTar(data []byte) (map[string]tarEntry, error) {
	reader := bytes.NewReader(data)
	var tr *tar.Reader
	var gz *gzip.Reader
	if len(data) >= 2 && data[0] == 0x1f && data[1] == 0x8b {
		var err error
		gz, err = gzip.NewReader(reader)
		if err != nil {
			return nil, err
		}
		defer gz.Close()
		tr = tar.NewReader(gz)
	} else {
		tr = tar.NewReader(reader)
	}

	entries := map[string]tarEntry{}
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if hdr.Typeflag != tar.TypeReg && hdr.Typeflag != tar.TypeRegA {
			continue
		}
		content, err := io.ReadAll(tr)
		if err != nil {
			return nil, err
		}
		isFile := false
		if hdr.PAXRecords != nil {
			if v, ok := hdr.PAXRecords["is_file"]; ok && v == "true" {
				isFile = true
			}
		}
		entries[hdr.Name] = tarEntry{data: content, isFile: isFile}
	}
	return entries, nil
}

func enrichTarError(out []byte, baseErr error) error {
	entries, err := readTar(out)
	if err != nil {
		return baseErr
	}
	if errMsg, ok := entries["error"]; ok {
		return fmt.Errorf(string(errMsg.data))
	}
	return baseErr
}

func exitWithError(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
