base=$1
query=$2
nt=$3
device=$4
start=$5
end=$6
step=$7
M=$8
is_ID=$9

bin=/home/zhengweiguo/liuchengjun/HybridNNS/build/hnns
log_path=/home/zhengweiguo/liuchengjun/HybridNNS/log/

echo "base: $base"

# for thr in $(seq 625 25 900)
for thr in $(seq $start $step $end)
do
    # echo "thr: $thr"
    # path=$log_path/$base.M_$M.efc_1000.efs_1000.ck_ts_0.ncheck_100.recall@1000.
    # # if [ "$is_ID" ]; then
    # #     path=$path"ID."
    # # fi
    # path=$path"thr_$thr.nthread_$nt.hnns.log"
    # if [ -f $path ]; then
    #     echo "$path exists"
    # else
    #     echo "$path does not exist, creating it"
    #     nohup $bin $base $query $M 1000 1000 $nt 1000 $thr hnns $device > $path
    # fi

    path=$log_path/$base.M_$M.efc_1000.efs_1000.ck_ts_500.ncheck_100.recall@1000.
    if [ "$is_ID" ]; then
        path=$path"ID."
    fi
    path=$path"thr_$thr.nthread_$nt.earlystop.log"
    if [ -f $path ]; then
        echo "$path exists"
    else
        echo "$path does not exist, creating it"
        nohup $bin $base $query $M 1000 1000 $nt 1000 $thr earlystop $device > $path
    fi

    # path=$log_path/$base.M_$M.efc_1000.efs_1000.ck_ts_0.ncheck_100.recall@1000.
    # # if [ "$is_ID" ]; then
    # #     path=$path"ID."
    # # fi
    # path=$path"thr_$thr.nthread_$nt.qonly.log"
    # if [ -f $path ]; then
    #     echo "$path exists"
    # else
    #     echo "$path does not exist, creating it"
    #     nohup $bin $base $query $M 1000 1000 $nt 1000 $thr hnns_qonly $device > $path
    # fi
done

path=$log_path/$base.M_$M.efc_1000.efs_1000.ck_ts_0.ncheck_100.recall@1000.nthread_$nt.random.log
if [ -f $path ]; then
    echo "$path exists"
else
    echo "$path does not exist, creating it"
    nohup $bin $base $query $M 1000 1000 $nt 1000 $thr random $device > $path
fi

# ../script/run.sh datacomp-image.base datacomp-text.query  96 7 690 690 1 32
# ../script/run.sh datacomp-text.base datacomp-image.query  96 7 400 400 1 32
# ../script/run.sh spacev100m.base spacev100m.query         96 7 775 775 1 32
# ../script/run.sh wikipedia.base wikipedia.query           96 7 960 960 1 32
# ../script/run.sh imagenet.base imagenet.query             96 7 990 990 1 32
# ../script/run.sh deep100m.base deep100m.query             96 7 975 975 1 32

# ../script/run.sh datacomp-text.base datacomp-image.query  48 7 650 650 1 32

# ../script/run.sh wikipedia.base wikipedia.query           48 7 960 960 1 128
