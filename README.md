# Data Availability Statement

This repo is the **Data Availability Statement** part of the manuscript entitled **"Design and Discovery of Novel Monopolar Spindle Kinase 1 (MPS1/TTK) Inhibitors by Computational Approaches"**

## Hardware and Software Requirements

### Hardware

 - **Machine**: Linux 4.18.0 x86_64, Red Hat 8.5.0
 - **CPUs**: Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz x 144
 - **GPU**: NVIDIA A100 40GB x 1, cuda version 11.6
 - **Memory**: 512GB
 - **Hard disk**: 10TB

### Software and Python package requirements

 - **Schrodinger**: Release 2020-3
 - **PaDEL**: 2.21
 - **Python**: 3.7.11
   - *rdkit*: 2020.09.1.0
   - *scikit-learn*: 0.24.2
   - *pytorch*: 1.11.0

### Appendix: Prepare Python Environment by Conda

 - create a environment named **production** by conda 4.9.2

```
(base) $ conda create -n production python=3.7.11
(base) $ conda activate production
(production) $ conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
(production) $ conda install -c rdkit rdkit=2020.09.1.0
(production) $ conda install -c conda-forge scikit-learn=0.24.2
(production) $ conda install matplotlib
(production) $ pip install cairosvg ipython paramiko
```

 - all the commands in this README are executed by the environment *production*

### Working Folder

Create a user in Linux machine named **cadd**, make a new folder named **paper_MPS1**, that is, the working folder is **/home/cadd/paper_MPS1/**. In this README file, all input, output, code, parameter, ... files are listed in this folder as the example.

## Experiment 1: MPS1 inhibitors collection and curation

 - make a new folder **data_curation**, and `cd` to it
 - manually prepare a file **S1_collect1556.csv**, contains 10 columns in it (NO, CompdKey, SMILES, Class, Type, Relation, Value, Unit, Source, Remark). *NO* ranged from 1 to 1556, *CompdKey* is the ID in paper or example ID in patent, *Type* is the enzymatic IC50 or cellular IC50, *Relation* is equal to, greater than, ..., *Source* is the Patent ID or ChEMBL Doc ID, *Remark* is the tag of high-ATP or low-ATP
 - select the *Class* as TTK, select *Remark* as low-ATP (assay with ATP 10 uM), remove n.d. and NA in *Value*, change all entries of *Unit* to nM. After above preprocessing, **1425** entries retained, add *Name* column assigned TTK_0001 to TTK_1425, save a file **S2_init1425.csv**
 - execute SMILES check by the script *check_smi_for_csv.py*:
   - input file **S2_init1425**, output file **S3_checked.csv**, all molecules passed

```
python /home/cadd/paper_MPS1/scripts/check_smi_for_csv.py --cur_folder /home/cadd/paper_MPS1/data_curation/ --in_file S2_init1425.csv --out_file S3_checked.csv --logfile check_smi_for_csv.log --smi_title SMILES --name_title Name --min_mw 150 --max_mw 800 --element_mode common
```

 - execute duplicate by the script *duplicate.py*:
   - input file **S3_checked.csv**, output file **S4a_unique.csv**, the duplicated infos are given in the *Duplicate* column
   - manually create a copy of **S4a_unique.csv** and rename as **S4b_unique_manual.csv**. For duplications, assign the median as the final IC50 value, add two columns *Unique_Value* and *Unique_Remark* to record the assigned reason. Seven molecules are assigned to delete because the large differences of duplications. The retained molecules are saved at **S4c_unique_remain.csv**, contains 1309 molecules.

```
python /home/cadd/paper_MPS1/scripts/duplicate.py --cur_folder /home/cadd/paper_MPS1/data_curation/ --in_csv S3_checked.csv --out_csv S4a_unique.csv --logfile duplicate.log --name_title Name --title_string Relation,Value
```

 - execute flatten duplicate by the script *duplicate_flatten.py*:
   - input file **S4c_unique_remain.csv**, output file **S4d_unique_flatten.csv**, the duplicated infos are given in the *Flatten* column
   - manually create a copy of **S4d_unique_flatten.csv** and rename as **S4e_unique_flatten_manual.csv**, add a column *Flatten_Value*, assign the best IC50 to the duplicated entries

```
python /home/cadd/paper_MPS1/scripts/duplicate_flatten.py --cur_folder /home/cadd/paper_MPS1/data_curated/--in_csv S4c_unique_remain.csv --out_csv S4d_unique_flatten.csv --logfile duplicate_flatten.log --name_title Name --title_string Unique_Value
```

 - assign highly or weakly actives:
   - create a copy of **S4e_unique_flatten_manual.csv** and rename as **S5a_assign_target_for_lig_model.csv**, add a column *target_binary*, which the *Flatten_Value* less than and equal to 10 nM as 1, otherwise as 0 (data with the *Flatten Value* 0-50 nM are tagged as not sure)
   - create a copy of **S4c_unique_remain.csv** and rename as **S5b_assign_target_for_rec_model.csv**, add a column *target_binary*, which the *Unique Value* less than and equal to 10 nM as 1, otherwise as 0  (data with the *Flatten Value* 0-50 nM are tagged as not sure)
   - remove the data tagged not sure, save output files **S6a_curated1178_for_lig.csv** and **S6b_curated1203_for_rec.csv**
 - *Note*: **S6a_curated1178_for_lig.csv** is the **Dataset 1** in the manuscript

## Experiment 2: Generate Decoys

 - current folder: /home/cadd/paper_MPS1/data_curation/
 - extract 741 highly actives in **S6b_curated1203_for_rec.csv** and save as **S7_extract_for_decoys.csv**
 - generate SDF file by the script *csv2sdf.py*:
   - input file **S7_extract_for_decoys.csv**, output file **S7_extract_for_decoys.sdf**

```
python /home/cadd/paper_MPS1/scripts/csv2sdf.py --cur_folder /home/cadd/paper_MPS1/data_curation/ --in_csv S7_extract_for_decoys.csv --smi_title Aromatic_SMILES --name_title Name --out_sdf S7_extract_for_decoys.sdf
```

 - execute molecular clustering by the script *cluster.py*:
   - cluster the 741 molecules to 50 clusters, and the most active one in each cluster are retained
   - input files **S7_extract_for_decoys.csv** and **S7_extract_for_decoys.sdf**, output file **S8_culster.csv**

```
python /home/cadd/paper_MPS1/scripts/cluster.py --cur_folder /home/cadd/paper_MPS1/data_curation/ --logfile cluster.log --in_csv S7_extract_for_decoys.csv --in_sdf S7_extract_for_decoys.sdf --out_csv S8_cluster.csv --n_clusters 50 --benchmark Unique_Value --keepmax N
```

 - submit to DUD-E website to obtain decoys:
   - manually prepare DUD-E input file **S8_cluster.smi**, which is modified from **S8_cluster.csv** (each line contains only `SMILES\tName`), submit the .SMI file to DUD-E website, provide an e-mail to receive the output file **dude-decoys.tar.gz**
   - unpacked the **dude-decoys.tar.gz**, manually summarize all 3100 decoys to a file **S9_decoys.csv**, assign names Decoy_0001 to Decoy_3100
 - check decoys SMILES by the script *check_smi_from_csv.py*:
   - input file **S9_decoys.csv**, output file **S9a_decoys_checked.csv**, 3097 decoys passed

```
python /home/cadd/paper_MPS1/scripts/check_smi_for_csv.py --cur_folder /home/cadd/paper_MPS1/data_curation/ --in_file S9_decoys.csv --out_file S9a_decoys_checked.csv --logfile decoys_checked.log --smi_title SMILES --name_title Name
```

 - execute duplicate by the script *duplicate.py*:
   - input file **S9a_decoys_checked.csv**, output file **S9b_decoys_unique.csv**, 3091 unique decoys retained

```
python /home/cadd/paper_MPS1/scripts/duplicate.py --cur_folder /home/cadd/paper_MPS1/data_curation/ --in_csv S9a_decoys_checked.csv --out_csv S9b_decoys_unique.csv --logfile decoys_unique.log --name_title Name
```

 - prepare the docking input file:
   - manually combine the files **S7_extract_for_decoys.csv** and **S9b_decoys_unique.csv** to a file **data_for_docking.csv**, keep columns *Name*, *Aromatic_SMILES*, *Kekulized_SMILES*, *InChiKey*, add a column *target_binary* that assign all actives as 1 and all decoys as 0 (3832 mols - 741 actives and 3091 decoys)

- *Note*: **data_for_docking.csv** is the **Dataset 2** in the manuscript

## Experiment 3: Ligand-based deep neural network binary classification models

 - make a new folder **ligand_model** and `cd` to it
 - copy file **S6a_curated1178_for_lig.csv** in folder *data_curation* to current folder
 - execute train/valid/test splitting by the script *model_utils_trtesplit.py*
   - input file **S6a_curated1178_for_lig.csv**, output file **data_trtecv.csv**

```
python /home/cadd/paper_MPS1/scripts/model_utils_trtesplit.py --cur_folder /home/cadd/paper_MPS1/ligand_model/ --in_csv S6a_curated1178_for_lig.csv --out_csv data_trtecv.csv --smi_title Aromatic_SMILES --target_title target_binary --out_trte_title trte --out_trtecv_title trtecv --radio 6 --valid 5
```

 - generate SDF file by the script *csv2sdf.py*:
   - input file **data_trtecv.csv**, output file **data2D.sdf**

```
python /home/cadd/paper_MPS1/scripts/csv2sdf.py --cur_folder /home/cadd/paper_MPS1/ligand_model/ --in_csv data_trtecv.csv --smi_title Aromatic_SMILES --name_title Name --out_sdf data2D.sdf
```

 - calculate molecular fingerprints and molecular descriptors by the scripts *cal_rdkit_mp.py* and *cal_padel_mp.py*:
   - input file **data2D.sdf**; output files **data2D_ECFP4.csv.gz**, **data2D_PubChemFP.csv.gz**, **data2D_RDKit2D.csv.gz**, and **data2D_PaDEL2D.csv.gz**
   - three addtional files suffixed with "_failedID.txt" are also generated, and these three files are empty, suggest that all molecular fingerprints/descriptors calculations are successful

```
python /home/cadd/paper_MPS1/scripts/cal_rdkit_mp.py --cur_folder /home/cadd/paper_MPS1/ligand_model/ --sdf_file data2D.sdf --mode ECFP4 --split_n 10
python /home/cadd/paper_MPS1/scripts/cal_rdkit_mp.py --cur_folder /home/cadd/paper_MPS1/ligand_model/ --sdf_file data2D.sdf --mode RDKit2D --split_n 10
python /home/cadd/paper_MPS1/scripts/cal_padel_mp.py --cur_folder /home/cadd/paper_MPS1/ligand_model/ --sdf_file data2D.sdf --mode PubChemFP --split_n 10
python /home/cadd/paper_MPS1/scripts/cal_padel_mp.py --cur_folder /home/cadd/paper_MPS1/ligand_model/ --sdf_file data2D.sdf --mode PaDEL2D --split_n 10
```

 - execute molecular fingerprint selection by the script *model_utils_select_fp_for_binary.py*:
   - input files **data2D_ECFP4.csv.gz** and **data2D_PubChemFP.csv.gz**, output files **feats_ECFP4.json** and **feats_PubChemFP.json**

```
python /home/cadd/paper_MPS1/scripts/model_utils_select_fp_for_binary.py --cur_folder /home/cadd/paper_MPS1/ligand_model/ --in_info_csv data_trtecv.csv --in_feats_csv data2D_ECFP4.csv.gz --in_failed_id N --out_json feats_ECFP4.json --trte_title trte --target_title target_binary --feat_start ECFP4_1 --feat_end ECFP4_1024 --keepnum_str 256,128
python /home/cadd/paper_MPS1/scripts/model_utils_select_fp_for_binary.py --cur_folder /home/cadd/paper_MPS1/ligand_model/ --in_info_csv data_trtecv.csv --in_feats_csv data2D_PubChemFP.csv.gz --in_failed_id N --out_json feats_PubChemFP.json --trte_title trte --target_title target_binary --feat_start PubChemFP_1 --feat_end PubChemFP_881 --keepnum_str 256,128
```

 - execute molecular descriptor selection by the script *model_utils_select_des_for_binary.py*:
   - input files **data2D_RDKit2D.csv.gz** and **data2D_PaDEL2D.csv.gz**, output files **feats_RDKit2D.json** and **feats_PaDEL2D.json**

```
python /home/cadd/paper_MPS1/scripts/model_utils_select_des_for_binary.py --cur_folder /home/cadd/paper_MPS1/ligand_model/ --in_info_csv data_trtecv.csv --in_feats_csv data2D_RDKit2D.csv.gz --in_failed _id N --out_json feats_RDKit2D.json --trte_title trte --feat_start BCUT2D_MWHI --feat_end fr_urea --ml_method dnn --pcc_inner 0.85 --min_de s_num 2 --max_des_num 32 --step_des_num 2
python /home/cadd/paper_MPS1/scripts/model_utils_select_des_for_binary.py --cur_folder /home/cadd/paper_MPS1/ligand_model/ --in_info_csv data_trtecv.csv --in_feats_csv data2D_PaDEL2D.csv.gz --in_failed_id N --out_json feats_PaDEL2D.json --trte_title trte --feat_start nAcid --feat_end Zagreb --ml_method dnn --pcc_inner 0.85 --min_des_num 2 --max_des_num 32 --step_des_num 2
```

 - build deep neural network models by the scripts *model_utils_build_dnn_for_binary.py*:
   - input files *data2D_XXX.csv.gz* and *feats_XXX.json*, output 912 modeling folders, each represent a model
   - a total of 38 commands are executed for building modeling, we prepare a shell script **build_dnn_models.sh** to summarize all commands in it, and execute at one time
   - all 912 models are too large, thus in this repo, we only keep the best four models

```
nohup bash build_dnn_models.sh > nohup_build_dnn_models.log 2>&1 &
```

 - summarize metrics of all 912 models, and save at **metrics.xlsx**, for each molecular feature, keep the model that has the best *va_MCC*, and finally, four models are retained
 - create the consensus model by the script *model_utils_consensus_for_dnn_binary.py*：
   - manually prepare a **consensus.json** file, output files **consensus.log**, **consensus.png**, and **consensus_score.txt**

```
python /home/cadd/paper_MPS1/scripts/model_utils_consensus_for_dnn_binary.py --cur_folder /home/cadd/paper_MPS1/ligand_model/ --info_file data_trtecv.csv --trtecv_title trtecv --target_title target_binary --des_prefix data2D --consensus_json consensus.json --logfile consensus.log --figfile consensus.png --summary_score_file consensus_score.txt --validnum 5 --device cpu
```

 - *Note*: the **four modeling folders** and the **consensus.json** make up the **ligand-based consensus model** in the manuscript; the **consensus.png** is the **Figure 3** in the manuscript

## Experiment 4: Consensus Docking Score

 - make a new folder named **docking_model** and `cd` to it
 - download five PDB files from Protein Data Bank database, i.e., **2x9e.pdb**, **4zeg.pdb**, **6h3k.pdb**, **6tnb.pdb**, and **6tnd.pdb**
 - open the *Maestro* software - *Protein Preparation Wizard* tool, take the 2x9e as an example, the five PDB files are preprocessed by the same way:
   - *Import structure from file* - *Browse* - **2x9e.pdb**
   - at *Import and Process* tab, select *Fill in missing side chains using Prime*, select *Fill in missing loops using Prime*, select *Cao termini*, click *Preprocess*
   - tranfer to *Refine* tab, select *Minimize hydrogens of altered species*, click *Optimize*, click *Remove waters*, click *Minimize*
   - return to *Maestro* interface, right-click the Entry suffixed with "minimized", click *export structure*, save as **2x9e_preprocess.pdb**
 - open *Receptor Grid Generation* tool, click any atom of the ligand, set the Job name as *glide-grid_2x9e*, click *Run*; output file **glide-grid_2x9e.zip**
 - use the same process to obtain the other four grid ZIP files
 - execute the *LigPrep* to obtain 3D conformation:
   - prepare the param file **ligprep.inp**, copy the **data_for_docking.csv** from the **data_curation** to current folder
   - generate **data_for_docking.sdf** by the script *csv2sdf.py*, then execute *ligprep* to generate the output file **data_for_docking_ligprep.maegz**

```
python /home/cadd/paper_MPS1/scripts/csv2sdf.py --cur_folder /home/cadd/paper_MPS1/docking_model/ --in_csv data_for_docking.csv --smi_title Aromatic_SMILES --name_title Name --out_sdf data_for_docking.sdf
$SCHRODINGER/ligprep -inp ligprep.inp -HOST localhost:12 -NJOBS 12
```

 - execute glide docking (take the 2x9e as the example):
   - make a sub-folder named **glide_2x9e** and `cd` to it
   - prepare two input files **Glide_SP.in** and **Glide_XP.in**
   - execute *glide*, obtain the output file **Glide_XP_pv.maegz**
   - execute *structconvert*, transfer the .MAEGZ file to .CSV file, obtain **Glide_XP_convert.csv**
   - match the *NAME* column of **Glide_XP_convert.csv** and the *Name* column of **data_for_docking.csv** by the script *model_utils_glide_matchname.py*, keep the only one entry in **Glide_XP_convert.csv** that has the best *r_i_docking_score*, obtain the output file **Glide_XP_unique.csv**

```
$SCHRODINGER/glide Glide_SP.in -HOST localhost:72 -NJOBS 72 -WAIT
$SCHRODINGER/glide Glide_XP.in -HOST localhost:72 -NJOBS 72 -WAIT
$SCHRODINGER/utilties/structconvert -n 2: Glide_XP_pv.maegz Glide_XP_convert.sdf
$SCHRODINGER/utilties/structconvert Glide_XP_convert.sdf Glide_XP_convert.csv
python /home/cadd/paper_MPS1/scripts/model_utils_glide_matchname.py --cur_folder /home/cadd/paper_MPS1/docking_model/glide_2x9e/ --dataset_csv /home/cadd/paper_MPS1/data_curation/data_for_docking.csv --dataset_name_title Name --dataset_target_title_str target_binary,Aromatic_SMILES,Kekulized_SMILES,InChiKey --glide_csv Glide_XP_convert.csv --glide_name_title NAME --out_match_csv Glide_XP_unique.csv --match_unique Y --unique_title r_i_docking_score --unique_keepmax N
```

 - the other four systems (4zeg, 6h3k, 6tnb, and 6tnd), execute the above same processes, each sub-folder has the output file **Glide_XP_unique.csv**
 - manually prepare a file **combine_manual.csv**, combine the five **Glide_XP_unique.csv** files, record the *target_binary* and *r_i_docking_score*, rename the docking score as the pdb code, add five new columns (*2x9e_rank*, *4zeg_rank*, *6h3k_rank*, *6tnb_rank*, and *6tnd_rank*), the top 20% of docking scores are assigned as 1 otherwise 0, summarized the five columns suffixed "_rank" and save at a new column *sum*
 - prepare a script *plot.py*, execute it, input file **combine_manual.csv**, output figure files **docking_score_hist.png** and **docking_score_roc.png**

```
python plot.py
```

 - *Note*: the column *sum* in **combine_manual.csv** is the **consensus docking score** in the manuscript; the figures **docking_score_hist.png** and **docking_score_roc.png** correspond to **Figure 4a** and **4b** in the manuscript

## Experiment 5: Pretrain the Recurrent Neural Network

 - make a new folder named **rnn_pretrain** and `cd` to it
 - download all ChEMBL compounds as .TSV format (assess 2022.12), extract only two columns *ChEMBL ID* and *Smiles* and save as **ChEMBL_allcompds.csv** (because the Github's file size limit of 100 MB, this file is uploaded as the packed format **ChEMBL_allcompds.7z** in the repo), contains 2,084,724 molecules
 - check smiles, flatten smiles, duplicate smiles, and remove smiles with pains group, obtain **ChEMBL_allcompds_unique.csv** (also uploaded as the packed format **ChEMBL_allcompds_unique.7z**), contains 1,727,489 molecules
 - prepare the pretrain script *pretrain_rnn.py* and execute it, input file **ChEMBL_allcompds_unique.csv**, output directory file **voc.json**, output model files **epochX_allstep13496.pth** (X = 1 ~ 5), output log file **pretrain.log**, output pretrain figure file **lossfig.png**

```
python pretrain_rnn.py
```

 - *Note*: the file **epoch5_allstep13496.pth** corresonds to the pretrained model in the manuscript; the **voc.json** corresponds to the vocalulary in the manuscript; the **lossfig.png** is the **Figure S1** in the manuscript

## Experiment 6: Fine-tune the Recurrent Neural Network

 - make a new folder named **rnn_fine_tune** and `cd` to it
 - copy **S7_extract_for_decoys.csv** to current folder
 - prepare an input file **highly_actives.csv** modified from the **S7_extract_for_decoys.csv**, add the *Flatten_SMILES* column
 - prepare a fine-tuning script *fine_tune.py* and execute it, input files **highly_actives.csv**, **voc.json**, and **epoch5_allstep13496.pth**; output model files **epoch_X.pth** (X = 1 ~ 15), output sampled SMILES files **epochX_sampled.smi** (X = 1 ~ 15), output log files **loss.log**, **tune.log**, and **unique.log**

```
python fine_tune.py
```

 - prepare a script *plot_loss.py* to obtain the loss figure during fine-tuning
   - input files **loss.log** and **unique.log**, output file **plot_loss.png**

```
python plot_loss.py
```

 - prepare a script *plot.py* to determine the optimal epoch of fine-tuning
   - input files **epochX_sampled.smi** (X = 1 ~ 15), output file **plot_5refs.png**

```
python plot.py
```

 - *Note*: the file **epoch_4.pth** corresponds to the final productive model in the manuscript; the files **plot_loss.png** and **plot_5refs.png** is the **Figure 5a** and **5b** in the manuscript, respectively

## Experiment 7: Molecular Generation and Convergence

 - create a new folder named **rnn_production** and `cd` to it
 - prepare a script **prod.py**, the input files **epoch_4.pth** and **voc.json**, the output file **genmols.csv**, contains 502,562 molecules

```
python prod.py
```

 - manually add *NO* column (from 1 to 502,562) and *Name* column (from RNN_0000001 to RNN_0502562) to the **genmols.csv**, save as **genmols_S1_init.csv**
 - execute smiles check by the script *check_smi_for_csv.py*:
   - input file **genmols_S1_init.csv**, output file **genmols_S2_checked.csv** (upload as the packed format **genmols_S2_checked.7z**), 495498 molecules passed

```
python /home/cadd/paper_MPS1/scripts/check_smi_for_csv.py --cur_folder /home/cadd/paper_MPS1/rnn_production/ --in_file genmols_S1_init.csv --out_file genmols_S2_checked.csv --logfile genmols_S2_checked.log --smi_title SMILES --name_title Name
```

 - generate SDF file by the script *csv2sdf.py*:
   - input **genmols_S2_checked.csv**, output **genmols.sdf** (1.39G, too large, not upload to the repo)

```
python /home/cadd/paper_MPS1/scripts/csv2sdf.py --cur_folder /home/cadd/paper_MPS1/rnn_production/ --in_csv genmols_S2_checked.csv --smi_title Aromatic_SMILES --name_title Name --out_sdf genmols.sdf
```

 - calculate molecular fingerprints/descriptors:
   - input file **genmols.sdf**
   - output files **genmols_XXX.csv.gz** (XXX = ECFP4, PubChemFP, RDKit2D, PaDEL2D), 1.21G, too large, not upload to the repo

```
python /home/cadd/paper_MPS1/scripts/cal_rdkit_mp.py --cur_folder /home/cadd/paper_MPS1/rnn_production/ --sdf_file genmols.sdf --mode ECFP4 --split_n 20
python /home/cadd/paper_MPS1/scripts/cal_rdkit_mp.py --cur_folder /home/cadd/paper_MPS1/rnn_production/ --sdf_file genmols.sdf --mode RDKit2D --split_n 20
python /home/cadd/paper_MPS1/scripts/cal_padel_mp.py --cur_folder /home/cadd/paper_MPS1/rnn_production/ --sdf_file genmols.sdf --mode PubChemFP --split_n 20
python /home/cadd/paper_MPS1/scripts/cal_padel_mp.py --cur_folder /home/cadd/paper_MPS1/rnn_production/ --sdf_file genmols.sdf --mode PaDEL2D --split_n 20
```

 - predict by the ligand-based consensus model:
   - input file **consensus.json**, output file **ligand_model_pred_results.csv** (contains 495498 molecules' results)

```
python /home/cadd/paper_MPS1/scripts/model_utils_predict_for_binary.py --cur_folder /home/cadd/paper_MPS1/rnn_production/ --consensus_json /home/cadd/paper_MPS1/ligand_model/consensus.json --out_csv ligand_model_pred_results.csv --device cpu --des_prefix genmols
```

 - predict by the consensus docking score:
   - execute *ligprep*, input file **genmols.sdf**, output file **genmols_ligprep.maegz** (1.41G, too large, not upload to the repo)
   - create a sub-folder named **glide_2x9e** and `cd` to it
   - execute glide three-tier docking by the script *glide_custom_noligprep.py*, output file **Glide_XP_convert.csv** (280M, too large, not upload to the repo)

```
$SCHRODINGER/ligprep -inp ligprep_genmols.inp -HOST localhost:144 -NJOBS 144
python /home/cadd/paper_MPS1/scripts/glide_custom_noligprep.py --cur_folder /home/cadd/paper_MPS1/rnn_production/glide_2x9e/ --gridfile glide-grid_2x9e.zip --ligprep_file /home/cadd/paper_MPS1/rnn_production/genmols_ligprep.maegz --keepnum_htvs 200000 --keepnum_sp 100000 --keepnum_xp 50000 --n_jobs 72
```

 - the other four systems (4zeg, 6h3k, 6tnb, and 6tnd) are executed by the same above processes to obtain the output file **Glide_XP_convert.csv**
 - manually create a file **docking_model_pred_results.csv** that combined the above five **Glide_XP_convert.csv** files, and calculate the consensus score save at the *Consensus* column
 - cluster the results to obtain the final results file **cluster_manual_picked.cdx**
 - *Note*: the molecules in **cluster_manual_picked.cdx** correspond to the **Figure S2** and **S3** in the manuscript

## Experiment 8: Molecular Dynamics Simulation

- create a new folder named **desmond_md** and `cd` to it
- create a new sub-folder named **build_system**, prepare three initial structures obtained from the Glide XP docking, that is **4zeg_CFI-402257_initial.pdb**, **4zeg_compd01_initial.pdb**, and **4zeg_compd10_initial.pdb**
- Build desmond MD solvent systems of the initial structures
  - prepare the input parameter file **desmond_system_build.msj**, containing the parameters of adding counterion ions, adding solvent box, assigning solvent model, assigning forcefield, and adding salts
  - transfer the input *.pdb* file to *.mae* file
  - execute system builder to obtain the desmond MD input file **desmond_setup_4zeg_CFI-402257-out.cms**
- the other two system files **desmond_setup_4zeg_compd01-out.cms** and **desmond_setup_4zeg_compd10-out.cms** can be obtained by the similar commands (change the input and output file names, job name)

```
mkdir build_system
cd build_system
# take 4zeg_CFI-402257_initial.pdb as an example
$SCHRODINGER/utilities/pdbconvert -ipdb 4zeg_CFI-402257_initial.pdb -omae 4zeg_CFI-402257_initial.mae
$SCHRODINGER/utilities/multisim -JOBNAME desmond_setup_4zeg_CFI-402257 -m desmond_system_build.msj 4zeg_CFI-402257_initial.mae -o desmond_setup_4zeg_CFI-402257-out.cms -HOST localhost
```

- create a new sub-folder named **4zeg_CFI-402257_500_seed2007**
  - folder meanings: the protein is 4zeg, ligand is CFI-402257, MD time is 500 ns, MD seed is 2007
  - copy the system building file to the current folder
  - prepare the parameter files **run_seed2007.cfg** and **run_seed2007.msj**, containing the parameters of minimization, heating, equilbration, production phases and simulation time
  - execute the *multisim* to obtain trajectory files and the output file **desmond-out.cms**
  - prepare the trajectory analysis input file **sid_in.eaf**, execute the script *analyze_simulation.py* to obtain output file **sid_out.eaf**
  - open *Maestro* software, *Simulation Interaction Diagram* tool, load the **sid_out.eaf** to obtain the trajectory analyses results, saved in **raw-data** and **images** folders
  - prepare the script *plot_rmsd.py* to obtain the plot of RMSD vs Time, input file **raw-data/PL_RMSD.dat**, output image **plot_rmsd.png**
- the other eight system files (all nine systems, that is, three ligands with three MD seeds) can be obtained by the similar commands
- *Note*: three **plot_rmsd.png** figures with seed 2007 corresponding to **Figure 7** in the manuscript
```
# Take the CFI-402257 (replica of seed 2007) as the example, the other eight simulations were same except for the folder and file names

# Make the simulation folder, cd to it, and copy the initial topology file to the folder
cd ..
mkdir 4zeg_CFI-402257_500_seed2007
cd 4zeg_CFI-402257_500_seed2007
cp ../build_system/desmond_setup_4zeg_CFI-402257-out.cms ./

# Manually prepare two parameter files (run_seed2007.msj and run_seed2007.cfj) and execute the simulations
$SCHRODINGER/utilities/multisim -JOBNAME desmond -HOST localhost -maxjob 10 -cpu 1 -m run_seed2007.msj -c run_seed2007.cfg desmond_setup_4zeg_CFI-402257-out.cms -mode unbrella -set 'stage[1].set_family.md.jlaunch_opt=["gpu"]' -o desmond_md_out.cms -lic DESMOND_GPGPU:16 -WAIT

# Manually prepare the input analysis file (sid_in.eaf) and execute the trajectory analyses
$SCHRODINGER/run analyze_simulation.py desmond-out.cms desmond_trj sid_out.eaf sid_in.eaf

# Plotting of RMSD vs Time
python ../plot_rmsd.py --dat_file raw-data/PL_RMSD.dat --out_file protlig_rmsd.png --dt 0.1 --minY 0 --maxY 5 --title_str "CFI-402257 (seed 2007)" --sel_prot_ca Y --sel_lig_fitby_prot Y
```
