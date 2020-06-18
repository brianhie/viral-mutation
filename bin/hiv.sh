for epoch in {01..04}
do
    CHECK_FNAME=target/hiv/checkpoints/bilstm/bilstm_512-$epoch.hdf5
    echo "Considering "$CHECK_FNAME"..."
    python bin/hiv.py bilstm --dim 512 --checkpoint $CHECK_FNAME --semantics
done
