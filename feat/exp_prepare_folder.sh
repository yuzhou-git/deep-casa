expname=deep_casa_wsj

basedir=$(pwd)

exp_dir=$basedir/exp/$expname

infeat_dir_tr=$basedir/exp/$expname/feat/tr
infeat_dir_cv=$basedir/exp/$expname/feat/cv
infeat_dir_tt=$basedir/exp/$expname/feat/tt

model_dir=$basedir/exp/$expname/models

output_tt_files=$basedir/exp/$expname/output_tt/files

mkdir -p $exp_dir $infeat_dir_tr $infeat_dir_tt $infeat_dir_cv $model_dir $output_tt_files 
