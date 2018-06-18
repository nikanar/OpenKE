test:
	echo Started testing $$(date) >> timing.log
	python2 test.py
	echo Finished testing $$(date) >> timing.log

train:
	echo Started training $$(date) >> timing.log
	python2 train.py
	echo Finished training $$(date) >> timing.log

recompile:
	./make.sh
