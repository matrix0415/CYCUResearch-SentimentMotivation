# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from os import listdir
from os.path import isfile
from os.path import abspath as abs
from os.path import join as pathjoin
from scipy import transpose as transarray
from multiprocessing import freeze_support
from multiprocessing import cpu_count as cpu
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from libs.nlpLib import nltkL as nltk
from libs.fileLib import fileRead as fread, fileWriteDict, fileReadDict
from libs.fileLib import fileWrite as fwrite
from libs.fileLib import fileReadLine as fline
from libs.fileLib import fileWriteLine as fwline

# GLOBAL SETTING
jobCapacity =cpu()*3                                # depend on your computer capacity.
nltkPath ="venv/nltk_data"                          # nltk_data location, if you want to use the default path, empty  it.
senticnetPath ="venv/nltk_data/senticnet3.rdf.xml"  # senticnet rdf file path.
nltk =nltk(nltkPath)                                # Object: libs.nltkL created.
tagList =[                                          # Part of speech tags you accept.
	"VB", "VBD", "VBN", "VBP", "VBZ",
	"NN", "NNS", "NNP", "NNPS",
	"JJ", "JJR", "JJS",
	"RB", "RBR", "RBS",
    "IN"
]

# Preprocess Config setting
stopwordList =[]                                    # Stopwords List-->['we', 'is'...]
preprocessDatasetLocation ="dataset/classified"     # Original File location.
preprocessTargetLocation ="dataset/filter"          # Opinions after pre-process.

# Calculate setting setting
opinionPerFile =1000                                # How many opinions you want to use to calculate the scores.
ngram =(1, 3)                                       # N-Gram
maxDF =0                                            # Maximum df number, delete if over.
normalization ='l2'                                 # TF-IDF normalization.
calTargetLocation ="dataset/calculate"              # Output Folder.
calSaveFList ="%s"%opinionPerFile+"-%ddf"%maxDF+"-%s-fList.csv"
calSaveOriFList ="%s"%opinionPerFile+"-%ddf"%maxDF+"-%s-original-fList.csv"
calSaveFeature ="%s"%opinionPerFile+"-%ddf"%maxDF+"-%s-feature.csv"
calSaveOriFeature ="%s"%opinionPerFile+"-%ddf"%maxDF+"-%s-original-feature.csv"
calSaveRs ="%s"%opinionPerFile+"-%ddf"%maxDF+"-%s-result.csv"
calSaveOriRs ="%s"%opinionPerFile+"-%ddf"%maxDF+"-%s-original-result.csv"
calSaveIdf ="%s"%opinionPerFile+"-%ddf"%maxDF+"-%s-idf.csv"
calSaveOriIdf ="%s"%opinionPerFile+"-%ddf"%maxDF+"-%s-original-idf.csv"
calSaveRsCombine ="%s"%opinionPerFile+"-%ddf"%maxDF+"-%s-combine.csv"

# Opinion Pick Setting
opinionPickPerFile =1000                           #

# Assign Score Setting
assignScoreLocation ="dataset/scores"
assignScoreScores ="%s-%ddf-%dperFile"%(opinionPerFile, maxDF, opinionPickPerFile)+"-%s-assign-scores.csv"
assignScorePolarity ="%s-%ddf-%dperFile"%(opinionPerFile, maxDF, opinionPickPerFile)+"-%s-assign-polarity.csv"
assignScoreFeatures ="%s-%ddf-%dperFile"%(opinionPerFile, maxDF, opinionPickPerFile)+"-%s-assign-features.csv"

# Classify Setting
cvFold =10
classifyJobs =5
resultFile ="dataset/result/%s-%ddf-%dperFile-%dfold"%(opinionPerFile, maxDF, opinionPickPerFile, cvFold)+"-%s"


def preprocessFunction(fname):
	rs =[]
	content =fread(pathjoin(preprocessDatasetLocation, fname))[1].split("\n")

	if content[0]:
		rs =([nltk.englishCorpusLemma(corpus, 0.8) for corpus in content])
		rs =(fname, [content for rsBool, content in rs if rsBool])
		print("%s\tFile Finished."%fname)

	else: print("%s\tFile Failed."%fname)

	return rs       #[[fName, content(list)], [fname, content(list)], ...]


def preprocessMain():
	from multiprocessing import Pool

	rs =[]
	freeze_support()
	fList =[i for i in sorted(listdir(preprocessDatasetLocation)) if i.split("_")[2] =="10"]
	targetList =sorted(listdir(preprocessTargetLocation))
	fList =[i for i in fList if i not in targetList]
	print("Preprocess-Process:\tpreprocessing")
	pool =Pool(processes =jobCapacity)
	rs =pool.map(preprocessFunction, fList)
	pool.close()
	pool.join()

	for fname, content in rs:
		print("Preprocess-Output:\t%s"%fname)
		outputBool, outputRs =fwrite(pathjoin(preprocessTargetLocation, fname), "\n".join(content))
		if outputBool: print("Preprocess-Output: Success.")
		else: print("Preprocess-Output: Failed.")

	return rs


def calculateDeleteArray(removeIndex, feature, rsArray, idf):
	removeIndex.reverse()

	for i in removeIndex:
		feature.pop(i)
		idf =np.delete(idf, i, 0)
		rsArray =np.delete(rsArray, i, 0)

	return (feature, rsArray, idf)


def calculateTFIDF(vec, corpus, que):
	trans =vec.fit_transform(corpus)
	feature =vec.get_feature_names()
	rsArray =transarray(trans.toarray())
	idf =vec.idf_
	print("Calculate-Process:\tTFIDF Finished, %d features."%len(feature))

	removeIndex =[index for index, content in enumerate([
		content for rsBool, content in [
			nltk.posTaggerFilter(str(word), acceptTagList =tagList) for word in feature
		]
	]) if content ==""]
	feature, rsArray, idf =calculateDeleteArray(removeIndex, feature, rsArray, idf)
	denyIdf =[key for key, value in enumerate(idf) if value<np.mean(idf)]
	feature, rsArray, idf =calculateDeleteArray(denyIdf, feature, rsArray, idf)
	length, width =rsArray.shape
	print("Calculate-Process:\tCalculating Finished, %dx%d."%(length, width))

	if length !=0 and width !=0:
		rsArray[rsArray < np.mean(rsArray)] =0

	que.put((feature, rsArray, idf))


def calculateMain(fcontent =[]):
	from multiprocessing import Process as process, Queue as queue

	rs =[]
	fList =[i for i in listdir(abs(preprocessTargetLocation)) if i.split("_")[2] =="10"]
	print("Calculate-Preparing:\tOpinion =%d, DF =%d"%(opinionPerFile, maxDF))

	if 	not isfile(pathjoin(calTargetLocation, calSaveOriRs%"token")) and not isfile(pathjoin(calTargetLocation, calSaveOriRs%"motivation")):
		if not fcontent or not fList:
			if not fcontent:
				fList.sort()
				fcontent =[
					(fname, content[1].split("\n")[:opinionPerFile]) for fname, content in (
						(fname, fread(pathjoin(preprocessTargetLocation, fname))) for fname in fList
					) if content[0]
				]

			tokenContent =[
				" ".join([" ".join(content) for fname, content in fcontent if fname.split("_")[0] =="business"]),
				" ".join([" ".join(content) for fname, content in fcontent if fname.split("_")[0] =="else"]),
				" ".join([" ".join(content) for fname, content in fcontent if fname.split("_")[0] =="family"]),
				" ".join([" ".join(content) for fname, content in fcontent if fname.split("_")[0] =="friends"]),
				" ".join([" ".join(content) for fname, content in fcontent if fname.split("_")[0] =="romance"]),
			]
			motivationContent =[
				" ".join([" ".join(content) for fname, content in fcontent if fname.split("_")[1] =="5.0"]),  #pos
				" ".join([" ".join(content) for fname, content in fcontent if fname.split("_")[1] =="3.0"]),  #obj
				" ".join([" ".join(content) for fname, content in fcontent if fname.split("_")[1] =="1.0"]),  #neg
			]

			freeze_support()
			tokenQue, motivationQue =queue(), queue()
			vec =tfidf(ngram_range =ngram, norm =normalization, use_idf =True, smooth_idf =False)
			if maxDF>=2: vec.max_df =maxDF
			jobs =[
				process(target=calculateTFIDF, args=[vec, tokenContent, tokenQue]),
				process(target=calculateTFIDF, args=[vec, motivationContent, motivationQue])
			]

			print("Calculate-Process:\tcalculating")
			for job in jobs: job.start()
			tokenOriFeature, tokenOriRsArray, tokenOriIdf =tokenQue.get()
			motivationOriFeature, motivationOriRsArray, motivationOriIdf =motivationQue.get()
			for job in jobs: job.join()

			print("Calculate-Output:\tOriginal File")
			column =['business', 'else', 'family', 'friends', 'romance']
			fwrite(pathjoin(calTargetLocation, calSaveOriFList%"token"), "\n".join(column))
			fwrite(pathjoin(calTargetLocation, calSaveOriFeature%"token"), "\n".join(tokenOriFeature))
			np.savetxt(pathjoin(calTargetLocation, calSaveOriIdf%"token"), tokenOriIdf, delimiter=",", fmt='%1.20f')
			np.savetxt(pathjoin(calTargetLocation, calSaveOriRs%"token"), tokenOriRsArray, delimiter=",", fmt='%1.20f')
			fwrite(pathjoin(calTargetLocation, calSaveOriFList%"motivation"), "\n".join(["pos", "obj", "neg"]))
			fwrite(pathjoin(calTargetLocation, calSaveOriFeature%"motivation"), "\n".join(motivationOriFeature))
			np.savetxt(pathjoin(calTargetLocation, calSaveOriIdf%"motivation"), motivationOriIdf, delimiter=",", fmt='%1.20f')
			np.savetxt(pathjoin(calTargetLocation, calSaveOriRs%"motivation"), motivationOriRsArray, delimiter=",", fmt='%1.20f')

	else:
		print("Calculate-Input:\tReading Files")
		tokenOriFeature =fread(pathjoin(calTargetLocation, calSaveOriFeature%"token"))[1].split("\n")
		tokenOriRsArray =np.genfromtxt(pathjoin(calTargetLocation, calSaveOriRs%"token"), delimiter=",")
		tokenOriIdf =np.genfromtxt(pathjoin(calTargetLocation, calSaveOriIdf%"token"), delimiter=",")
		motivationOriFeature =fread((calTargetLocation, calSaveOriFeature%"motivation"))[1].split("\n")
		motivationOriRsArray =np.genfromtxt(pathjoin(calTargetLocation, calSaveOriRs%"motivation"), delimiter=",")
		motivationOriIdf =np.genfromtxt(pathjoin(calTargetLocation, calSaveOriIdf%"motivation"), delimiter=",")

	return (tokenOriFeature, tokenOriRsArray, tokenOriIdf, motivationOriFeature, motivationOriRsArray, motivationOriIdf)


def transformMain(data):
	if not isfile(pathjoin(calTargetLocation, calSaveRs%"final")) and not isfile(pathjoin(calTargetLocation, calSaveFeature%"final")):
		if not data:
			tokenOriFeature =fread(pathjoin(calTargetLocation, calSaveOriFeature%"token"))[1].split("\n")
			tokenOriRsArray =np.genfromtxt(pathjoin(calTargetLocation, calSaveOriRs%"token"), delimiter=",")
			tokenOriIdf =np.genfromtxt(pathjoin(calTargetLocation, calSaveOriIdf%"token"), delimiter=",")
			motivationOriFeature =fread((calTargetLocation, calSaveOriFeature%"motivation"))[1].split("\n")
			motivationOriRsArray =np.genfromtxt(pathjoin(calTargetLocation, calSaveOriRs%"motivation"), delimiter=",")
			motivationOriIdf =np.genfromtxt(pathjoin(calTargetLocation, calSaveOriIdf%"motivation"), delimiter=",")

		else:
			tokenOriFeature, tokenOriRsArray, tokenOriIdf ,motivationOriFeature, motivationOriRsArray, motivationOriIdf =data

		print("Calculate-Process:\tcomparing")
		compareList =[(key, tokenOriFeature.index(word)) for key, word in enumerate(motivationOriFeature) if word in tokenOriFeature]
		compareListMotivation =[i for i, j in compareList]
		compareListToken =[j for i, j in compareList]
		motivationDelList =[i for i in range(len(motivationOriFeature)) if i not in compareListMotivation]
		tokenDelList =[i for i in range(len(tokenOriFeature)) if i not in compareListToken]

		print("Calculate-Process:\tmatching")
		motivationFeature, motivationRsArray, motivationIdf=calculateDeleteArray(
			motivationDelList, motivationOriFeature, motivationOriRsArray, motivationOriIdf
		)
		tokenFeature, tokenRsArray, tokenIdf=calculateDeleteArray(
			tokenDelList, tokenOriFeature, tokenOriRsArray, tokenOriIdf
		)

		if motivationFeature ==tokenFeature:
			print("Calculate-Process:\tmultiplying")
			finalFeature =motivationFeature
			tmp1 =motivationRsArray
			tmp1 =np.delete(tmp1, 1, 1)
			tmp1 *=10000
			tmp1[:,1] *=-1
			tmp2 =tokenRsArray.T*tmp1.T[0]
			tmp3 =tokenRsArray.T*tmp1.T[1]
			finalRsArray =np.vstack((tmp2, tmp3)).T

			print("Calculate-Output:\ttoken")
			column =['business', 'else', 'family', 'friends', 'romance']
			fwrite(pathjoin(calTargetLocation, calSaveFList%"token"), "\n".join(column))
			fwrite(pathjoin(calTargetLocation, calSaveFeature%"token"), "\n".join(tokenFeature))
			np.savetxt(pathjoin(calTargetLocation, calSaveIdf%"token"), tokenIdf, delimiter=",", fmt='%1.20f')
			np.savetxt(pathjoin(calTargetLocation, calSaveRs%"token"), tokenRsArray, delimiter=",", fmt='%1.20f')
			print("Calculate-Output:\tmotivation")
			fwrite(pathjoin(calTargetLocation, calSaveFList%"motivation"), "\n".join(["pos", "neg"]))
			fwrite(pathjoin(calTargetLocation, calSaveFeature%"motivation"), "\n".join(motivationFeature))
			np.savetxt(pathjoin(calTargetLocation, calSaveIdf%"motivation"), motivationIdf, delimiter=",", fmt='%1.20f')
			np.savetxt(pathjoin(calTargetLocation, calSaveRs%"motivation"), motivationRsArray, delimiter=",", fmt='%1.20f')
			print("Calculate-Output:\tfinal")
			fwrite(pathjoin(calTargetLocation, calSaveFList%"final"), "\n".join(["pos", "neg"]))
			fwrite(pathjoin(calTargetLocation, calSaveFeature%"final"), "\n".join(finalFeature))
			np.savetxt(pathjoin(calTargetLocation, calSaveRs%"final"), finalRsArray, delimiter=",", fmt='%1.7f')

		else:
			print("Calculate-Output:\tError!")
			fwrite(pathjoin(calTargetLocation, "error"+calSaveOriFeature%"token"), "\n".join(tokenFeature))
			fwrite(pathjoin(calTargetLocation, "error"+calSaveOriFeature%"motivation"), "\n".join(motivationFeature))

	else:
		print("Transform-Input:\tReading Files")
		finalFeature =fread(pathjoin(calTargetLocation, calSaveFeature%"final"))[1].split("\n")
		finalRsArray =np.genfromtxt(pathjoin(calTargetLocation, calSaveRs%"final"), delimiter=",")

	return (finalFeature, finalRsArray)


def scoreCombineMain(rsarray):
	return (np.mean(rsarray, axis=1), rsarray.max(axis=1))


def opinionPickMain(**kwargs):
	import itertools

	fcontent =[]
	flist =[(float(i.split("_")[1]), pathjoin(preprocessTargetLocation, i))
	        for i in listdir(preprocessTargetLocation) if i.split("_")[1] =="1.0" or i.split("_")[1] =="5.0"]

	if 'count' in kwargs:
		totalOpinion =kwargs['count']
		print("Opinion Pick-Picking:\ttotal %d, %d/per file"%(totalOpinion, totalOpinion/len(flist)))
		tmp =[fline(path =f, count =totalOpinion/len(flist))[1] for p, f in flist if p ==5.0]
		pos =[(0, i) for i in list(itertools.chain(*tmp))]
		tmp =[fline(path =f, count =totalOpinion/len(flist))[1] for p, f in flist if p ==1.0]
		neg =[(1, i) for i in list(itertools.chain(*tmp))]

	elif 'line' in kwargs:
		line =kwargs['line']
		print("Opinion Pick-Picking:\ttotal %d, %d/per file"%((line[1]-line[0])*len(flist), line[1]-line[0]))
		tmp =[fline(path =f, line =(line[0], line[1]))[1] for p, f in flist if p ==5.0]
		pos =[(0, i[1]) for i in list(itertools.chain(*tmp)) if i[0]]
		tmp =[fline(path =f, line =(line[0], line[1]))[1] for p, f in flist if p ==1.0]
		neg =[(1, i[1]) for i in list(itertools.chain(*tmp)) if i[0]]

	return pos+neg


# assignScoreFunction(token, sentimentScoreLibObj =OBJ, scoreType ="polarity")
def assignScoreFunction(token, **kwargs):
	rs =0.0

	if 'sentimentScoreLibObj' and 'scoreType' in kwargs:
		rsbool, rs =kwargs['sentimentScoreLibObj'].fetchScore(token =token, tag =kwargs['scoreType'])
		if not rsbool: print(rs)

	elif 'feature' and 'score' in kwargs:
		if token in kwargs['feature']:
			rs =kwargs['score'][kwargs['feature'].index(token)]

	return rs


# assignScoreMain(opinionData, lexicon ='senticwordnet', rdfPath ='PATH', tags =['polarity'], name ="String")
# assignScoreMain(opinionData, feature =[], score =np.array([]), name ="String")
def assignScoreMain(opinionData, **kwargs):
	from sklearn.feature_extraction.text import CountVectorizer as vectorizer
	from libs.nlpLib import sentimentScoreLib as sentiscore

	scoreList =[]

	if True:#not isfile(pathjoin(assignScoreLocation, assignScoreScores%kwargs['name'])):
		polarityTag =np.array([polarity for polarity, i in opinionData], dtype ='bool_')
		opinions =[nltk.posTaggerFilter(str(i), acceptTagList =tagList) for polarity, i in opinionData]
		opinions =[content for rsbool, content in opinions if rsbool]
		vec =vectorizer(ngram_range =(1,3), dtype =float)
		rsarray =vec.fit_transform(opinions)
		feature =vec.get_feature_names()
		bywords =rsarray.toarray().T
		print("Assigning Score-Process:\tassigning, %s"%kwargs['name'])

		if 'lexicon' in kwargs:
			senticnetObj =sentiscore(type=kwargs['lexicon'], rdfPath=kwargs['rdfPath'], tags=kwargs['tags'], nltkPath=nltkPath)
			scoreList =[assignScoreFunction(token=w, sentimentScoreLibObj=senticnetObj, scoreType="polarity") for w in feature]

		elif 'feature' and 'score' in kwargs:
			scoreList =[assignScoreFunction(token=w, feature=kwargs['feature'], score=kwargs['score']) for w in feature]

		else:
			print("Assigning Score-Process:\tError, none of the options.")

		for index, s in enumerate(scoreList): bywords[index:index+1] *=s
		resultArray =bywords.T

		print("Assigning Score-Output:\tfinal")
		fwrite(pathjoin(assignScoreLocation, assignScoreFeatures%kwargs['name']), "\n".join(feature))
		np.savetxt(pathjoin(assignScoreLocation, assignScorePolarity%kwargs['name']), polarityTag, delimiter=",", fmt='%1.1i')
		np.savetxt(pathjoin(assignScoreLocation, assignScoreScores%kwargs['name']), resultArray, delimiter=",", fmt='%1.7f')

	else:
		print("Assigning Score-Input:\tReading Files")
		feature =fread(pathjoin(assignScoreLocation, assignScoreFeatures%kwargs['name']))[1].split("\n")
		polarityTag =np.genfromtxt(pathjoin(assignScoreLocation, assignScorePolarity%kwargs['name']), delimiter=",")
		resultArray =np.genfromtxt(pathjoin(assignScoreLocation, assignScoreScores%kwargs['name']), delimiter=",")

	return (feature, resultArray, polarityTag)


def classifyMain(name ="", rsarray =np.array([]), polarityTag =np.array([])):
	from sklearn import svm, cross_validation as cv

	if not isfile(resultFile%name):
		model = svm.SVC(verbose =True, probability =True)
		accuracy =cv.cross_val_score(model, rsarray, polarityTag, scoring='accuracy', n_jobs =classifyJobs, cv =cvFold)
		avgPrecision =cv.cross_val_score(model, rsarray, polarityTag, scoring='average_precision', n_jobs =classifyJobs, cv =cvFold)
		precision =cv.cross_val_score(model, rsarray, polarityTag, scoring='precision', n_jobs =classifyJobs, cv =cvFold)
		recall =cv.cross_val_score(model, rsarray, polarityTag, scoring='recall', n_jobs =classifyJobs, cv =cvFold)
		f1 =cv.cross_val_score(model, rsarray, polarityTag, scoring='f1', n_jobs =classifyJobs, cv =cvFold)
		rs ={"accuracy": accuracy, "avgPrecision": avgPrecision, "f1": f1, "precision": precision, "recall": recall}
		fileWriteDict(resultFile%name, rs)

	else:
		rs =fileReadDict(resultFile%name)

	return rs


if __name__ =="__main__":
	start ="\n%s Started.----------------------"

	print(start%"Preprocess")
	data =preprocessMain()

	print(start%"Calculate")
	rsCalculate =calculateMain(data)

	print(start%"Transform")
	feature, rsarray =transformMain(rsCalculate)

	print(start%"Combining Scores")
	avgRs, maxRs =scoreCombineMain(rsarray)

	print(start%"Opinion Pick")
	content =opinionPickMain(count =opinionPickPerFile)

	print(start%"Assigning Score")
	feature, rsSentic, tags =assignScoreMain(content, lexicon ='senticnet', name ="senticnet", rdfPath =senticnetPath, tags=['polarity'])
	print(start%"Classifying")
	rs =classifyMain(name='senticnet', rsarray=rsSentic, polarityTag=tags)
	print(rs)

	print(start%"Assigning Score")
	feature, rsAvgRs, tags =assignScoreMain(content, feature =feature, score =avgRs, name ="average")
	print(start%"Classifying")
	rs =classifyMain(name ="average", rsarray=rsAvgRs, polarityTag=tags)
	print(rs)

	print(start%"Assigning Score")
	feature, rsMaxRs, tags =assignScoreMain(content, feature =feature, score =maxRs, name ="max")
	print(start%"Classifying")
	rs =classifyMain(name ="max", rsarray=rsMaxRs, polarityTag=tags)
	print(rs)