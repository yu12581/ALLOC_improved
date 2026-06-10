$PYTHON = "C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe"
$BASE = "runs\full_40ep_bestauc"
$LOG  = "runs\full_40ep_bestauc_run.log"

New-Item -ItemType Directory -Force -Path $BASE | Out-Null
"" | Out-File $LOG -Encoding utf8

$seeds  = @(42, 1337, 2026)
$digits = 0..9
$total  = $seeds.Count * $digits.Count
$n = 0; $skip = 0; $fail = @()

foreach ($seed in $seeds) {
    foreach ($digit in $digits) {
        $n++
        $out = "$BASE\d${digit}_s${seed}"
        if (Test-Path "$out\summary.json") {
            Write-Host "[$n/$total] SKIP d${digit}_s${seed}"
            $skip++; continue
        }
        Write-Host "[$n/$total] digit=$digit seed=$seed"
        $cmd = "mnist_experiment.py --variant alocc_loss --specific $digit --epochs 40 --seed $seed --train-count 4096 --batch-size 64 --out-per-class-count 300 --noise-std 0.31 --r-alpha 0.2 --d-outclass-loss-scale 0.1 --selection-strategy best_auc --output-dir $out"
        $p = Start-Process -FilePath $PYTHON -ArgumentList $cmd -NoNewWindow -Wait -PassThru
        if ($p.ExitCode -eq 0) {
            "OK d${digit}_s${seed}" | Out-File $LOG -Append -Encoding utf8
        } else {
            Write-Warning "FAILED d${digit}_s${seed} (exit $($p.ExitCode))"
            "FAIL d${digit}_s${seed}" | Out-File $LOG -Append -Encoding utf8
            $fail += "d${digit}_s${seed}"
        }
    }
}

Write-Host "=== 完成 $n/$total  跳过 $skip  失败 $($fail.Count) ==="
if ($fail.Count -gt 0) { Write-Warning ($fail -join ", ") }
