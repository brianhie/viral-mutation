bash bin/benchmark_sweep_h1.sh > benchmark_escape_h1.log 2>&1 &

bash bin/benchmark_sweep_h3.sh > benchmark_escape_h3.log 2>&1 &

bash bin/benchmark_sweep_hiv.sh > benchmark_escape_hiv.log 2>&1 &

bash bin/benchmark_sweep_cov2rbd.sh > benchmark_escape_cov2rbd.log 2>&1 &

wait

echo "H1" > escape_results.log
python bin/plot_sweep.py benchmark_escape_h1.log >> escape_results.log 2>&1

echo -e "\nH3" >> escape_results.log
python bin/plot_sweep.py benchmark_escape_h3.log >> escape_results.log 2>&1

echo -e "\nHIV" >> escape_results.log
python bin/plot_sweep.py benchmark_escape_hiv.log >> escape_results.log 2>&1

echo -e "\nCoV-2 RBD" >> escape_results.log
python bin/plot_sweep.py benchmark_escape_cov2rbd.log >> escape_results.log 2>&1
