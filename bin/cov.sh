for file in target/cov/checkpoints/bilstm/*
do
    echo "Considering "$file"..."
    CUDA_VISIBLE_DEVICES=5 python bin/cov.py bilstm --checkpoint $file --semantics
done
