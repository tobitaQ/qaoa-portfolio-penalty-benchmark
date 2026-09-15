# GPU queue after E1 (2026-09-11)

`gpu_queue.sh` waits for the tmux session `e1` to end, then runs, in order:

1. re-times the 10 N=12 rows listed in `e1_rerun_ids.txt` (their `runtime_s`
   was measured while two extra processes briefly shared the GPU);
2. E3-A: ADAM step 0.03, 100 steps, on instances 0-9 x seeds 42-46 x {Aheur, Amargin};
3. E3-B: L-BFGS, 100 gradient evaluations, same block.

The seed extension (seeds 47-51 on the full cross, ~14 GPU h) was taken out
on 2026-09-11: the unit of generalisation is the instance (plan §10.1), so
more seeds per instance sharpen only the within-instance tail estimate. It
is listed with its command in the project's deferred-experiments list, to be added if
the E1 contrasts or a reviewer call for it.

Every step resumes, so the script can be started again at any time:

    tmux new -s queue "bash experiments/paper01_qubo_baseline/queue/gpu_queue.sh 2>&1 | tee -a logs/gpu_queue.log"
