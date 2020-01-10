dist_doc_DATA = UserManual.md

if HAVE_DOXYGEN

htmlpkg  = $(PACKAGE_NAME)-$(PACKAGE_VERSION)-html.tar.gz
doc_DATA = $(htmlpkg)
docstamp = doc.stamp

$(htmlpkg): $(docstamp)
	tar chof - html | gzip -9 -c > $@

$(docstamp): Doxyfile
	$(DOXYGEN)       $<
	echo Timestamp > $@

install-data-hook:
	cd $(DESTDIR)$(docdir) && tar xf $(htmlpkg)

uninstall-hook:
	cd $(DESTDIR)$(docdir) && $(RM) -fr html

CLEANFILES = $(docstamp) $(htmlpkg)

clean-local:
	$(RM) -fr html

endif
