base=$1
query=$2
nt=$3
device=$4
start=$5
end=$6
step=$7

bin=/home/zhengweiguo/liuchengjun/HybridNNS/build/hnns
log_path=/home/zhengweiguo/liuchengjun/HybridNNS/log/

echo "base: $base"

# for thr in $(seq 625 25 900)
for thr in $(seq $start $step $end)
do
    echo "thr: $thr"
    # if os.path.exists(log_path) is True, then pass
    path=$log_path/$base.M_32.efc_1000.efs_1000.ck_ts_0.ncheck_100.recall@1000.thr_$thr.nthread_$nt.hnns.log
    if [ -f $path ]; then
        echo "$path exists"
    else
        echo "$path does not exist, creating it"
        nohup $bin $base $query 32 1000 1000 $nt 1000 $thr hnns $device > $path
    fi

    path=$log_path/$base.M_32.efc_1000.efs_1000.ck_ts_0.ncheck_100.recall@1000.thr_$thr.nthread_$nt.qonly.log
    if [ -f $path ]; then
        echo "$path exists"
    else
        echo "$path does not exist, creating it"
        nohup $bin $base $query 32 1000 1000 $nt 1000 $thr hnns_qonly $device > $path
    fi
done

path=$log_path/$base.M_32.efc_1000.efs_1000.ck_ts_0.ncheck_100.recall@1000.nthread_$nt.random.log
if [ -f $path ]; then
    echo "$path exists"
else
    echo "$path does not exist, creating it"
    nohup $bin $base $query 32 1000 1000 $nt 1000 $thr random $device > $path
fi

# ../script/run.sh datacomp-image.base datacomp-text.query  48 7 690 690 1
# ../script/run.sh datacomp-text.base datacomp-image.query  48 7 400 400 1
# ../script/run.sh spacev100m.base spacev100m.query         48 7 775 775 1
# ../script/run.sh wikipedia.base wikipedia.query           48 7 960 960 1
# ../script/run.sh imagenet.base imagenet.query             48 7 990 990 1
# ../script/run.sh deep100m.base deep100m.query             48 7 975 975 1