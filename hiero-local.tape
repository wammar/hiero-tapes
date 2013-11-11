#!/usr/bin/env ducttape

global {
  ducttape_experimental_submitters=enable
  ducttape_experimental_imports=enable
}

import ../submitters.tape

task Tokenize
    < corpus=(DataSection:
                train=$train_corpus
                tune=$tune_corpus
                test=$test_corpus)
    > tokenized_corpus
    :: tokenize_corpus=@
    :: cdec_dir=@
#    :: .submitter=torque_shared .walltime="01:00:00" .cpus=1 .vmem=2g .q=shared 
{
  if [[ $tokenize_corpus == "false" ]]; then
    cp $corpus $tokenized_corpus
  else
    cat $corpus | $cdec_dir/corpus/tokenize-anything.sh > $tokenized_corpus
  fi
}

task Lowercase
    < corpus=$tokenized_corpus@Tokenize
    > lowercased_corpus
    :: lowercase_corpus=@
    :: cdec_dir=@
#    :: .submitter=torque_shared .walltime="01:00:00" .cpus=1 .vmem=2g .q=shared 
{
  if [[ $lowercase_corpus == "false" ]]; then
    cp $corpus $lowercased_corpus
  else
    cat $corpus | $cdec_dir/corpus/lowercase.pl > $lowercased_corpus
  fi
}

task FilterCorpus
    < corpus=$lowercased_corpus@Lowercase[DataSection:train]
    > filtered_corpus
    :: max_sentence_length=@
    :: cdec_dir=@
   # :: .submitter=torque_shared .walltime="01:00:00" .cpus=1 .vmem=2g .q=shared 
{
  if (( $max_sentence_length > 0 )); then
    $cdec_dir/corpus/filter-length.pl -$max_sentence_length $corpus > $filtered_corpus
  else
    cp $corpus $filtered_corpus
  fi
}

task FastAlignS2T
    < corpus=$filtered_corpus@FilterCorpus
    > alignment
    :: cdec_dir=@
    #:: .submitter=torque_shared .walltime="12:00:00" .cpus=1 .vmem=32g .q=shared 
{
  $cdec_dir/word-aligner/fast_align -i $corpus -d -v -o > $alignment
}

task FastAlignT2S
    < corpus=$filtered_corpus@FilterCorpus
    > alignment
    :: cdec_dir=@
   # :: .submitter=torque_shared .walltime="12:00:00" .cpus=1 .vmem=32g .q=shared
 {
  $cdec_dir/word-aligner/fast_align -i $corpus -d -v -o -r > $alignment
}

task GizaAlignS2T
    < corpus=$filtered_corpus@FilterCorpus
    > alignment
    > output_dir
    :: giza_bin=@
    :: moses_train_script=@
    :: giza_cores=@
    :: src=@
    :: tgt=@
    :: cdec_dir=@
  #  :: .submitter=torque_normal .walltime="12:00:00" .cpus=1 .vmem=32g .q=normal 
{

  $cdec_dir/corpus/cut-corpus.pl 1 $corpus > $corpus.$src
  $cdec_dir/corpus/cut-corpus.pl 2 $corpus > $corpus.$tgt

  $moses_train_script --first-step 1 --last-step 3 --parallel --root-dir $output_dir --corpus $corpus --f $src --e $tgt --external-bin-dir $giza_bin --mgiza $giza_bin --mgiza-cpus=$giza_cores --alignment srctotgt 
  cp $output_dir/model/aligned.srctotgt $alignment
}

task GizaAlignT2S
    < corpus=$filtered_corpus@FilterCorpus
    < output_dir=$output_dir@GizaAlignS2T
    > alignment
    :: giza_bin=@
    :: moses_train_script=@
    :: giza_cores=@
    :: src=@
    :: tgt=@
    :: cdec_dir=@
  #  :: .submitter=torque_normal .walltime="12:00:00" .cpus=32 .vmem=32g .q=normal
 {

  $cdec_dir/corpus/cut-corpus.pl 1 $corpus > $corpus.$src
  $cdec_dir/corpus/cut-corpus.pl 2 $corpus > $corpus.$tgt

  $moses_train_script --first-step 3 --last-step 3 --parallel --root-dir $output_dir --corpus $corpus --f $src --e $tgt --external-bin-dir $giza_bin --mgiza $giza_bin --mgiza-cpus=$giza_cores --alignment tgttosrc 
  cp $output_dir/model/aligned.tgttosrc $alignment
}

task GenerateWordpairFeatsS2T 
    :: data_file=$train_corpus 
    :: mkcls_bin=@
    > wordpair_feats_file 
    #:: .submitter=torque_shared .walltime="12:00:00" .cpus=32 .vmem=32g .q=normal 
{

  # use cdec to generate word pair features for a partiuclar test set
  ~/cdec/word-aligner/aligner.pl --mkcls=$mkcls_bin $data_file
  pushd talign
  make generate-wordpair-features
  gzip -d grammars/wordpairs.f-e.features.gz
  popd
  
  cp "talign/grammars/wordpairs.f-e.features" wordpair_feats_file
} 

task GenerateWordpairFeatsT2S 
    :: data_file=$train_corpus 
    :: mkcls_bin=@
    :: wammar_utils_dir=@
    :: tgt=@
    :: src=@
    > wordpair_feats_file 
    #:: .submitter=torque_normal .walltime="12:00:00" .cpus=32 .vmem=64g .q=normal 
{

  
  python $wammar_utils_dir/horizontal-split-parallel-corpus.py -i $data_file -1 src -2 tgt
  python $wammar_utils_dir/paste.py -1 tgt -2 src -d " ||| " -o reverse.$tgt-$src

  # use cdec to generate word pair features for a partiuclar test set
  ~/cdec/word-aligner/aligner.pl --mkcls=$mkcls_bin reverse.$tgt-$src
  pushd talign
  make generate-wordpair-features
  gzip -d grammars/wordpairs.f-e.features.gz
  popd
  
  cp "talign/grammars/wordpairs.f-e.features" wordpair_feats_file
  
} 

task BuildLatentCrfAligner
    :: alignment_with_openfst_dir=@
    > executable
   # :: .submitter=torque_shared .walltime="12:00:00" .cpus=1 .vmem=1g .q=shared 
{

  pushd $alignment_with_openfst_dir
  make -f Makefile-latentCrfAligner
  cp train-latentCrfAligner $executable
  popd
}

task AutoencoderAlignS2T
    :: alignment_with_openfst_dir=@
    :: test_sents_count=@ 
    :: procs=$giza_cores
    :: data_file=$train_corpus
    :: init_theta=$init_theta_fwd
    :: init_lambda=$init_lambda_fwd
    :: l2_strength=@
    :: l1_strength=@
    :: dirichlet_alpha=@
    :: variational=@
    < executable=$executable@BuildLatentCrfAligner
    < wordpair_feats_file=$wordpair_feats_file@GenerateWordpairFeatsS2T 
    > alignment 
    > out_err 
   # :: .submitter=torque_normal .walltime="48:00:00" .cpus=32 .vmem=64g .q=normal 
{

  # the latent-CRF word alignment mode
  command="mpirun -np $procs $executable
  --train-data $data_file 
  --output-prefix prefix 
  --feat SRC0_TGT0 --feat LOG_ALIGNMENT_JUMP --feat DIAGONAL_DEVIATION --feat SYNC_START --feat SYNC_END 
  --feat ALIGNMENT_JUMP --feat ALIGNMENT_JUMP_IS_ZERO --feat PRECOMPUTED 
  --wordpair-feats $wordpair_feats_file 
  --test-size $test_sents_count 
  --max-em-iter-count 1 
  --max-lbfgs-iter-count 1"

  if [ $init_theta ]; then
    command="$command --init-theta $init_theta"  
  fi

  if [ $init_lambda ]; then
    command="$command --init-lambda $init_lambda"
  fi

  if [ $l2_strength ]; then
    command="$command --l2-strength $l2_strength"
  fi

  if [ $l1_strength ]; then
    command="$command --l1-strength $l1_strength"
  fi

  if [ $dirichlet_alpha ]; then
    command="$command dirichlet-alpha $dirichlet_alpha"
  fi

  if [ $variational ]; then
    command="$command --variational-inference true"
  fi

  echo "executing $command..."
  $command 2> $out_err


  cp prefix.labels $alignment

}

task AutoencoderAlignT2S
    :: test_sents_count=@ 
    :: procs=$giza_cores
    :: data_file=$train_corpus
    :: init_theta=$init_theta_bwd
    :: init_lambda=$init_lambda_bwd
    :: l2_strength=@
    :: l1_strength=@
    :: dirichlet_alpha=@
    :: variational=@
    < wordpair_feats_file=$wordpair_feats_file@GenerateWordpairFeatsT2S 
    < executable=$executable@BuildLatentCrfAligner
    > alignment 
    > out_err 
   # :: .submitter=torque_normal .walltime="48:00:00" .cpus=32 .vmem=64g .q=normal 
{

  # the latent-CRF word alignment mode
  command="mpirun -np $procs $executable 
  --train-data $data_file 
  --output-prefix prefix 
  --feat SRC0_TGT0 --feat LOG_ALIGNMENT_JUMP --feat DIAGONAL_DEVIATION --feat SYNC_START --feat SYNC_END 
  --feat ALIGNMENT_JUMP --feat ALIGNMENT_JUMP_IS_ZERO --feat PRECOMPUTED 
  --wordpair-feats $wordpair_feats_file 
  --test-size $test_sents_count 
  --max-em-iter-count 1 
  --max-lbfgs-iter-count 1 
  --reverse true"

  if [ $init_theta ]; then
    command="$command --init-theta $init_theta"
  fi

  if [ $init_lambda ]; then
    command="$command --init-lambda $init_lambda"
  fi 

  if [ $l2_strength ]; then
    command="$command --l2-strength $l2_strength"
  fi

  if [ $l1_strength ]; then
    command="$command --l1-strength $l1_strength"
  fi

  if [ $dirichlet_alpha ]; then
    command="$command dirichlet-alpha $dirichlet_alpha"
  fi

  if [ $variational ]; then
    command="$command --variational-inference true"
  fi

  echo "executing $command..."
  $command  2> $out_err

  cp prefix.labels $alignment
}

task Symmetrize
    < s2tAlignment=(Aligner:
                      giza=$alignment@GizaAlignS2T
                      fast=$alignment@FastAlignS2T
                      autoencoder=$alignment@AutoencoderAlignS2T
                      autoencoder_giza=$alignment@AutoencoderAlignS2T
                      autoencoder_fast=$alignment@AutoencoderAlignS2T
                      autoencoder_none=$alignment@AutoencoderAlignS2T
                      fast_autoencoder=$alignment@FastAlignS2T
                      giza_autoencoder=$alignment@GizaAlignS2T)
    < t2sAlignment=(Aligner:
                      giza=$alignment@GizaAlignT2S
                      fast=$alignment@FastAlignT2S
                      autoencoder=$alignment@AutoencoderAlignT2S
                      autoencoder_giza=$alignment@GizaAlignT2S
                      autoencoder_fast=$alignment@FastAlignT2S
                      autoencoder_none=$alignment@AutoencoderAlignS2T
                      fast_autoencoder=$alignment@AutoencoderAlignT2S
                      giza_autoencoder=$alignment@AutoencoderAlignT2S)
    > alignment
    :: cdec_dir=@
   # :: .submitter=torque_shared .walltime="01:00:00" .cpus=1 .vmem=2g .q=shared 
{
  $cdec_dir/utils/atools -i $s2tAlignment -j $t2sAlignment -c grow-diag-final-and > $alignment
}

task BuildSuffixArray
     < corpus=$filtered_corpus@FilterCorpus
     < alignment=$alignment@Symmetrize  
     > ini
     > suffix_array
    :: cdec_dir=@
#    :: .submitter=torque_shared .walltime="48:00:00" .cpus=1 .vmem=33g .q=shared 
{
  export PYTHONPATH=`echo $cdec_dir/python/build/lib.*`
  python -m cdec.sa.compile -b $corpus -a $alignment -c $ini -o $suffix_array
}

task ExtractGrammars
    < corpus=(ExtractSection:
                tune=$lowercased_corpus@Lowercase[DataSection:tune]
                test=$lowercased_corpus@Lowercase[DataSection:test])
    < ini=@BuildSuffixArray
    > wrapped_corpus
    > grammar_dir
    :: cores=1
    :: cdec_dir=@
   # :: .submitter=torque_shared .walltime="24:00:00" .cpus=1 .vmem=33g .q=shared 
{
  export PYTHONPATH=`echo $cdec_dir/python/build/lib.*`
  cat $corpus | python -m cdec.sa.extract -c $ini -g $grammar_dir -j $cores -z > $wrapped_corpus
}

task BuildLanguageModel
    < corpus=$filtered_corpus@FilterCorpus
    < lm_data=@
    > language_model
    :: lm_order=@
    :: cdec_dir=@
  #  :: .submitter=torque_shared .walltime="06:00:00" .cpus=1 .vmem=33g .q=shared 
{
  corpus_target="./corpus_target"
  $cdec_dir/corpus/cut-corpus.pl 2 $corpus > $corpus_target
  cat $corpus_target $lm_data | $cdec_dir/klm/lm/builder/builder --order $lm_order > $language_model
}

task CompileLanguageModel
    < arpa=$language_model@BuildLanguageModel
    > language_model
    :: cdec_dir=@
 #   :: .submitter=torque_shared .walltime="06:00:00" .cpus=1 .vmem=33g .q=shared 
{
  $cdec_dir/klm/lm/build_binary $arpa $language_model
}

task MakeCdecIni
    < language_model=(UseCustomLM:
                        no=$language_model@CompileLanguageModel
                        yes=$language_model)
    > cdec_ini {
  echo "formalism=scfg" > $cdec_ini
  echo "add_pass_through_rules=true" >> $cdec_ini
  echo "feature_function=WordPenalty" >> $cdec_ini
  echo "feature_function=KLanguageModel $language_model" >> $cdec_ini
}

task Tune
    < tune_set=$wrapped_corpus@ExtractGrammars[ExtractSection:tune]
    < cdec_ini=@MakeCdecIni
    > mira_work
    > optimized_weights
    :: tune_corpus=@
    :: cores=$decode_cores
    :: cdec_dir=@
   # :: .submitter=torque_normal .walltime="48:00:00" .cpus=32 .vmem=60g .q=normal 
{


  ref_count=$(head -n 1 $tune_set | grep -o '|||' | wc -l)
  #cut -f 1 -d '|' $tune_set > ./tune_source
  for i in `seq 1 $ref_count`; do
     cut -f $(expr 3 \* $i + 1) -d '|' $tune_set > ./tune_refs.$i
  done

  initial_weights="./initial_weights"
  echo "SampleCountF 0.0"      >> $initial_weights 
  echo "CountEF 0.0"           >> $initial_weights
  echo "MaxLexEgivenF 0.0"     >> $initial_weights
  echo "EgivenFCoherent 0.0"   >> $initial_weights
  echo "MaxLexFgivenE 0.0"     >> $initial_weights
  echo "PassThrough 0.0"       >> $initial_weights
  echo "PassThrough_1 0.0"     >> $initial_weights
  echo "PassThrough_2 0.0"     >> $initial_weights
  echo "PassThrough_3 0.0"     >> $initial_weights
  echo "PassThrough_4 0.0"     >> $initial_weights
  echo "PassThrough_5 0.0"     >> $initial_weights
  echo "PassThrough_6 0.0"     >> $initial_weights
  echo "IsSingletonF 0.0"      >> $initial_weights
  echo "IsSingletonFE 0.0"     >> $initial_weights
  echo "LanguageModel 0.0"     >> $initial_weights
  echo "LanguageModel_OOV 0.0" >> $initial_weights
  echo "Glue 0.0"              >> $initial_weights
  echo "WordPenalty 0.0"       >> $initial_weights

  $cdec_dir/training/mira/mira.py -j $cores --update-size 250 -o $mira_work --weights ./initial_weights \
    --devset $tune_set --config $cdec_ini

  index=$(ls $mira_work | grep '^weights\.[0-9]*$' | sed 's/^weights\.//' | sort -n | tail -n 1)
  ln -s $mira_work/weights.$index $optimized_weights
}

task Decode
    < test_set=(TuneOrTest:
        test=$wrapped_corpus@ExtractGrammars[ExtractSection:test]
        tune=$wrapped_corpus@ExtractGrammars[ExtractSection:tune])
    < cdec_ini=@MakeCdecIni
    < weights=$optimized_weights@Tune
    > kbest
    > output
    :: k=1000
    :: cores=$decode_cores
    :: cdec_dir=@
 #   :: .submitter=torque_normal .walltime="6:00:00" .cpus=32 .vmem=60g .q=normal 
{

  cat $test_set | $cdec_dir/corpus/cut-corpus.pl 1 | $cdec_dir/decoder/cdec -c $cdec_ini -w $weights -k $k -r > $kbest
  sed 's/ ||| /\t/g' $kbest | sort -u -k 1,1 -n | cut -f 2 > $output
}

task Evaluate 
    < output=$output@Decode
    < refs=(TuneOrTest: test=$test_corpus tune=$tune_corpus)
    > bleu meteor ter length
    :: multeval=@
    :: meteor_lang=@
   # :: .submitter=torque_shared .walltime="00:20:00" .cpus=1 .vmem=4g .q=shared 
{
  METEORTASK=li
  METEORLANG=$meteor_lang
  scoreFile=scores.txt

  num_refs=$(head -n 1 $refs | grep -o '|||' | wc -l)
  for i in `seq 1 $num_refs`; do
    cut -f $(expr 3 \* $i + 1) -d '|' $refs | sed 's/^\s*//' | sed 's/\s*$//' > refs.$i
  done

  ln -s $(dirname $multeval)/constants .
  echo "executing: $multeval eval --refs refs.* --hyps-baseline $output --meteor.task $METEORTASK --meteor.language $METEORLANG &> $scoreFile"
  $multeval eval --refs refs.* --hyps-baseline $output --meteor.task $METEORTASK --meteor.language $METEORLANG &> $scoreFile
  echo "done executing that command."
  tail -n 2 $scoreFile | head -n 1 | sed 's/(\S\+)//g' | sed 's/\s\+/\t/g' | cut -f 2 > $bleu
  tail -n 2 $scoreFile | head -n 1 | sed 's/(\S\+)//g' | sed 's/\s\+/\t/g' | cut -f 3 > $meteor
  tail -n 2 $scoreFile | head -n 1 | sed 's/(\S\+)//g' | sed 's/\s\+/\t/g' | cut -f 4 > $ter
  tail -n 2 $scoreFile | head -n 1 | sed 's/(\S\+)//g' | sed 's/\s\+/\t/g' | cut -f 5 > $length
}

task EvaluateAlignment
    :: conv_pharaoh_script=@
    :: aer_eval_script=@
    :: gold_alignment=@
    :: test_sents_count=@ 
    < alignment=$alignment@Symmetrize 
    > aer 
#    :: .submitter=torque_shared .walltime="00:20:00" .cpus=1 .vmem=4g .q=shared 
  {
  head -n $test_sents_count $alignment | $conv_pharaoh_script > temp
  $aer_eval_script    temp    $gold_alignment     > $aer
}

#summary EvaluationSummary {
#  of Evaluate > Score {
#    cp $bleu $Score
#  }
#}
