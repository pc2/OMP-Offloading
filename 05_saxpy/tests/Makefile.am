if HAVE_CCSALLOC
cmdrun = ${CCSALLOC}
else
cmdrun = `which bash`
endif

check_SCRIPTS = saxpy_test_00.sh

TESTS = $(check_SCRIPTS)

saxpy_test_00.sh: $(top_srcdir)/src/saxpy
	echo "${cmdrun} saxpy_real_00.sh" > $@
	chmod +x $@

CLEANFILES = saxpy_test_00.sh
