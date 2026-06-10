$PYTHON = "C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe"
$RUNNER = "mnist_experiment.py"
$BASE   = "runs\ablation_cbam_s1"
$LOG    = "runs\ablation_cbam_s1_b.log"

New-Item -ItemType Directory -Force -Path $BASE | Out-Null
"" | Out-File $LOG -Encoding utf8

$DIGITS = @(0, 2, 8)
$SEEDS  = @(42, 2026)
$total  = $DIGITS.Count * $SEEDS.Count
$n = 0; $failed = @()

$COMMON = "--variant alocc_loss --epochs 40 --train-count 4096 --batch-size 64 " +
          "--out-per-class-count 300 --noise-std 0.31 --r-alpha 0.2 " +
          "--d-outclass-loss-scale 0.1 --selection-strategy best_auc"

foreach ($seed in $SEEDS) {
    foreach ($digit in $DIGITS) {
        $n++
        $out = "${BASE}\B_s1_d${digit}_s${seed}"
        if (Test-Path "${out}\summary.json") {
            Write-Host "[$n/$total] SKIP B_s1_d${digit}_s${seed}"
            continue
        }
        Write-Host "[$n/$total] B_s1  digit=${digit}  seed=${seed}"
        "B_s1_d${digit}_s${seed}" | Out-File $LOG -Append -Encoding utf8

        $cmd = "$RUNNER $COMMON --specific $digit --seed $seed --output-dir $out " +
               "--bottleneck-rank 8 --bottleneck-dropout 0.3"
        $p = Start-Process -FilePath $PYTHON -ArgumentList $cmd -NoNewWindow -Wait -PassThru
        if ($p.ExitCode -eq 0) {
            "OK" | Out-File $LOG -Append -Encoding utf8
        } else {
            Write-Warning "FAILED B_s1_d${digit}_s${seed}"
            "FAIL" | Out-File $LOG -Append -Encoding utf8
            $failed += "B_s1_d${digit}_s${seed}"
        }
    }
}

Write-Host "=== 完成 $n/$total  失败 $($failed.Count) ==="
if ($failed.Count -gt 0) { Write-Warning ($failed -join ", ") }
