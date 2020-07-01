for epoch in {02..20}
do
    CHECK_FNAME=target/covall/checkpoints/bilstm/bilstm_512-$epoch.hdf5
    echo "Considering "$CHECK_FNAME"..."
    CUDA_VISIBLE_DEVICES=2 python bin/sarscov2.py bilstm --dim 512 --checkpoint $CHECK_FNAME --semantics --namespace covall
done
