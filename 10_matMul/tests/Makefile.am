if HAVE_CCSALLOC
cmdrun = ${CCSALLOC}
else
cmdrun = `which bash`
endif

check_SCRIPTS = matMul_test_00.sh

TESTS = $(check_SCRIPTS)

matMul_test_00.sh: $(top_srcdir)/src/matMul
	echo "${cmdrun} matMul_real_00.sh" > $@
	chmod +x $@

CLEANFILES = matMul_test_00.sh
