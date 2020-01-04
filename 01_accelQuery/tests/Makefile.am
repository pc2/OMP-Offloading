if HAVE_CCSALLOC
cmdrun = ${CCSALLOC}
else
cmdrun = `which bash`
endif

check_SCRIPTS = accelQuery_test_00.sh

TESTS = $(check_SCRIPTS)

accelQuery_test_00.sh: $(top_srcdir)/src/accelQuery
	echo "${cmdrun} accelQuery_real_00.sh" > $@
	chmod +x $@

CLEANFILES = accelQuery_test_00.sh
