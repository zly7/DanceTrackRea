## 生成视频
python ./tools/txt2video_dance.py --img_path ../dataset/dancetrack_data --split train2 --gt

Linux 下生成图片，注意这里面代码有是否生成视频这个whether变量，请及时修改

python ./tools/txt2video_dance.py --img_path /home/zly/multi_ob/data/DanceTrack --split val --gt

python ./tools/txt2video_dance.py --img_path /home/zly/multi_ob/data/DanceTrack --split val --tracker yolox_x_from_online

python ./tools/txt2video_dance.py --img_path /home/zly/multi_ob/data/DanceTrack --split test --tracker yolox_x_from_online

## 快速生成结果
CUDA_VISIBLE_DEVICES=1 python3 ByteTrack/tools/track.py -f ByteTrack/exps/example/dancetrack/yolox_x.py -c ByteTrack/from_author/bytetrack_model.pth.tar -b 1 -d 1 --fp16 --fuse



python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL val  --METRICS HOTA CLEAR Identity  --GT_FOLDER /home/zly/multi_ob/data/DanceTrack/val --SEQMAP_FILE dancetrack/val_seqmap.txt --SKIP_SPLIT_FOL True   --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True --NUM_PARALLEL_CORES 8 --PLOT_CURVES False --TRACKERS_FOLDER /home/zly/multi_ob/DanceTrack/YOLOX_outputs/yolox_x/track_results

## 启动可视化前端
cd /home/zly/multi_ob/DanceTrack/vis_react/dance-track-viewer
PORT=7788 npm start