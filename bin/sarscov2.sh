for epoch in {04..20}
do
    CHECK_FNAME=target/covall/checkpoints/bilstm/bilstm_512-$epoch.hdf5
    echo "Considering "$CHECK_FNAME"..."
    CUDA_VISIBLE_DEVICES=2 python bin/cov.py bilstm --dim 512 --checkpoint $CHECK_FNAME --semantics --namespace covall
done
