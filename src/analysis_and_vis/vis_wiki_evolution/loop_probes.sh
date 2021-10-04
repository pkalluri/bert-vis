# Bash script to loop through both bert and gpt and a bunch of filters

for analysis in 'about' 'in'
do 
	if [[ analysis == 'in' ]]; then
		analysisflag=--probe_tok
	elif [[ analysis == 'about' ]]; then
		analysisflag=''
	fi

	for model in gpt bert
	do 
		small=big-data/wiki-small
		if [[ model == gpt ]]; then
			corpusdir=big-data/gpt-wiki-large/
			partial_filter=''
		elif [[ model == bert ]]; then
			corpusdir=big-data/wiki-large/
			partial_filter=bert-partial
		fi

		for filter in top random $partial_filter
		do
			params="-k 0 -l 1 -n 1000 -c 10"

			job --mem=50G -c 4 --wrap="/u/nlp/anaconda/main/anaconda3/envs/ria-bert-vis/bin/python src/analysis_and_vis/vis_wiki_evolution/probe_folk_wisdom.py $small info-$analysis-tok.csv $filter $params $analysisflag" -J ${model:0:1}-$analysis-${filter}
		done

	done
done