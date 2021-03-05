mkdir -p logs
mkdir -p results

for method in "greedy" "top" "tocp"
do
    for H in 2 4 6 8 10
    do
        for seed in {1..120}
        do
            python main.py $method $seed $H > logs/$method-H$H-s$seed.txt
        done
    done
done

python summary.py

python plot.py
