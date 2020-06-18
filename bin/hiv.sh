for epoch in {01..07}
do
    CHECK_FNAME=target/hiv/checkpoints.old/train_all/bilstm/bilstm_512-$epoch.hdf5
    echo "Considering "$CHECK_FNAME"..."
    python bin/hiv.py bilstm --dim 512 --checkpoint $CHECK_FNAME --semantics
done
