if HAVE_CCSALLOC
cmdrun = ${CCSALLOC}
else
cmdrun = `which bash`
endif

check_SCRIPTS = scalarAddition_test_00.sh

TESTS = $(check_SCRIPTS)

scalarAddition_test_00.sh: $(top_srcdir)/src/scalarAddition
	echo "${cmdrun} scalarAddition_real_00.sh" > $@
	chmod +x $@

CLEANFILES = scalarAddition_test_00.sh
