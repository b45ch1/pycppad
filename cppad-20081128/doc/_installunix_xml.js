var list_across0 = [
'_contents_xml.htm',
'_reference.xml',
'_index.xml',
'_search_xml.htm',
'_external.xml'
];
var list_up0 = [
'cppad.xml',
'install.xml',
'installunix.xml'
];
var list_down2 = [
'install.xml',
'introduction.xml',
'ad.xml',
'adfun.xml',
'library.xml',
'preprocessor.xml',
'example.xml',
'appendix.xml'
];
var list_down1 = [
'installunix.xml',
'installwindows.xml'
];
var list_down0 = [
'subversion.xml'
];
var list_current0 = [
'installunix.xml#Fedora',
'installunix.xml#RPM',
'installunix.xml#Download',
'installunix.xml#Download.Subversion',
'installunix.xml#Download.Web Link',
'installunix.xml#Download.Unix Tar Files',
'installunix.xml#Download.Tar File Extraction',
'installunix.xml#Configure',
'installunix.xml#Testing Return Status',
'installunix.xml#PrefixDir',
'installunix.xml#--with-Documentation',
'installunix.xml#--with-Introduction',
'installunix.xml#--with-Introduction.get_started',
'installunix.xml#--with-Introduction.exp_apx',
'installunix.xml#--with-Example',
'installunix.xml#--with-TestMore',
'installunix.xml#--with-Speed',
'installunix.xml#--with-Speed.cppad',
'installunix.xml#--with-Speed.double',
'installunix.xml#--with-Speed.profile',
'installunix.xml#--with-Speed.example',
'installunix.xml#--with-PrintFor',
'installunix.xml#--with-stdvector',
'installunix.xml#PostfixDir',
'installunix.xml#AdolcDir',
'installunix.xml#AdolcDir.Fix Adolc',
'installunix.xml#AdolcDir.Linux',
'installunix.xml#AdolcDir.Cygwin',
'installunix.xml#FadbadDir',
'installunix.xml#SacadoDir',
'installunix.xml#BoostDir',
'installunix.xml#IpoptDir',
'installunix.xml#CompilerFlags',
'installunix.xml#make',
'installunix.xml#make install'
];
function choose_across0(item)
{	var index          = item.selectedIndex;
	item.selectedIndex = 0;
	if(index > 0)
		document.location = list_across0[index-1];
}
function choose_up0(item)
{	var index          = item.selectedIndex;
	item.selectedIndex = 0;
	if(index > 0)
		document.location = list_up0[index-1];
}
function choose_down2(item)
{	var index          = item.selectedIndex;
	item.selectedIndex = 0;
	if(index > 0)
		document.location = list_down2[index-1];
}
function choose_down1(item)
{	var index          = item.selectedIndex;
	item.selectedIndex = 0;
	if(index > 0)
		document.location = list_down1[index-1];
}
function choose_down0(item)
{	var index          = item.selectedIndex;
	item.selectedIndex = 0;
	if(index > 0)
		document.location = list_down0[index-1];
}
function choose_current0(item)
{	var index          = item.selectedIndex;
	item.selectedIndex = 0;
	if(index > 0)
		document.location = list_current0[index-1];
}
