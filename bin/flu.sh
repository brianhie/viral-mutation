for epoch in {01..20}
do
    CHECK_FNAME=target/flu/checkpoints/dnn/dnn_512-$epoch.hdf5
    echo "Considering "$CHECK_FNAME"..."
    python bin/flu.py dnn --dim 512 --checkpoint $CHECK_FNAME --semantics
done
