for epoch in {09..20}
do
    CHECK_FNAME=target/cov/checkpoints.old/train_all/bilstm/bilstm_256-09.hdf5
    echo "Considering "$CHECK_FNAME"..."
    python bin/sarscov2.py bilstm --dim 256 --checkpoint $CHECK_FNAME --semantics --namespace covold
done
