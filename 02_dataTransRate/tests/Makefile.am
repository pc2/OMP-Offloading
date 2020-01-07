if HAVE_CCSALLOC
cmdrun = ${CCSALLOC}
else
cmdrun = `which bash`
endif

check_SCRIPTS = dataTransRate_test_00.sh

TESTS = $(check_SCRIPTS)

dataTransRate_test_00.sh: $(top_srcdir)/src/dataTransRate
	echo "${cmdrun} dataTransRate_real_00.sh" > $@
	chmod +x $@

CLEANFILES = dataTransRate_test_00.sh
