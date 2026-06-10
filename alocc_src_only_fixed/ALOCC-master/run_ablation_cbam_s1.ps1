$PYTHON = "C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe"
$RUNNER = "mnist_experiment.py"
$BASE   = "runs\ablation_cbam_s1"
$LOG    = "runs\ablation_cbam_s1.log"

New-Item -ItemType Directory -Force -Path $BASE | Out-Null
"" | Out-File $LOG -Encoding utf8

$DIGITS = @(0, 2, 8)
$SEEDS  = @(42, 2026)
$total  = $DIGITS.Count * $SEEDS.Count * 4
$n = 0; $failed = @()

$COMMON = "--variant alocc_loss --epochs 40 --train-count 4096 --batch-size 64 " +
          "--out-per-class-count 300 --noise-std 0.31 --r-alpha 0.2 " +
          "--d-outclass-loss-scale 0.1 --selection-strategy best_auc"

function Run-One($tag, $digit, $seed, $extra) {
    $script:n++
    $out = "${BASE}\${tag}_d${digit}_s${seed}"
    if (Test-Path "${out}\summary.json") {
        Write-Host "[$($script:n)/$total] SKIP ${tag}_d${digit}_s${seed}"
        return
    }
    Write-Host "[$($script:n)/$total] ${tag}  digit=${digit}  seed=${seed}"
    "${tag}_d${digit}_s${seed}" | Out-File $LOG -Append -Encoding utf8

    $cmd = "$RUNNER $COMMON --specific $digit --seed $seed --output-dir $out $extra"
    $p = Start-Process -FilePath $PYTHON -ArgumentList $cmd -NoNewWindow -Wait -PassThru
    if ($p.ExitCode -eq 0) {
        "OK" | Out-File $LOG -Append -Encoding utf8
    } else {
        Write-Warning "FAILED ${tag}_d${digit}_s${seed}"
        "FAIL" | Out-File $LOG -Append -Encoding utf8
        $script:failed += "${tag}_d${digit}_s${seed}"
    }
}

foreach ($seed in $SEEDS) {
    foreach ($digit in $DIGITS) {
        # A: 无 S1，无 CBAM（对照基线）
        Run-One "A_base"    $digit $seed ""
        # B: S1 only（rank=8, dropout=0.3）
        Run-One "B_s1"      $digit $seed "--bottleneck-rank 8 --bottleneck-dropout 0.3"
        # C: CBAM only
        Run-One "C_cbam"    $digit $seed "--use-cbam"
        # D: S1 + CBAM
        Run-One "D_s1_cbam" $digit $seed "--bottleneck-rank 8 --bottleneck-dropout 0.3 --use-cbam"
    }
}

Write-Host "=== 完成 $n/$total  失败 $($failed.Count) ==="
if ($failed.Count -gt 0) { Write-Warning ($failed -join ", ") }
