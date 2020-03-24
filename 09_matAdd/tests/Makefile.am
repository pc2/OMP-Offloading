if HAVE_CCSALLOC
cmdrun = ${CCSALLOC}
else
cmdrun = `which bash`
endif

check_SCRIPTS = matAdd_test_00.sh

TESTS = $(check_SCRIPTS)

matAdd_test_00.sh: $(top_srcdir)/src/matAdd
	echo "${cmdrun} matAdd_real_00.sh" > $@
	chmod +x $@

CLEANFILES = matAdd_test_00.sh
