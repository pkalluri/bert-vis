# Bash script to loop through both bert and gpt and a bunch of filters

small=true
concat=false

filetag=""
if [[ $small == true ]]; then 
	filetag="-small"
	params="-k 3 -n 1000 -c 10"
else
	params="-k 5 -n 10000 -c 100"
fi
if [[ $concat == true ]]; then
	jobtag="${filetag}-concat"
	params="$params -l 1 --concatenate -1"
fi


for analysis in 'about'
do 
	if [[ $analysis == 'in' ]]; then
		analysisflag=--probe_tok
	elif [[ $analysis == 'about' ]]; then
		analysisflag=''
	fi

	for model in gpt bert
	do 
		debugging_corpus=big-data/wiki-small
		if [[ $model == gpt ]]; then
			corpusdir=big-data/gpt-wiki-large/
			partial=''
		elif [[ $model == bert ]]; then
			corpusdir="big-data/wiki-large/"
			partial=bert-partial
		fi

		for filter in top random $partial NN JJ VB RB
		do			
			job --mem=50G -c 4 --wrap="/u/nlp/anaconda/main/anaconda3/envs/ria-bert-vis/bin/python src/analysis_and_vis/vis_wiki_evolution/probe_folk_wisdom.py $corpusdir info-$analysis-tok${filetag}.csv $filter $params $analysisflag" -J ${model:0:1}-$analysis-${filter}${jobtag}
		done

	done
done