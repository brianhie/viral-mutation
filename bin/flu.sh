for epoch in {02..20}
do
    CHECK_FNAME=target/flu/checkpoints/bilstm/bilstm_512-$epoch.hdf5
    echo "Considering "$CHECK_FNAME"..."
    python bin/flu.py bilstm --dim 512 --checkpoint $CHECK_FNAME --semantics
done
