var list_across0 = [
'_contents.htm',
'_reference.htm',
'_index.htm',
'_search.htm',
'_external.htm'
];
var list_up0 = [
'cppad.htm',
'install.htm',
'installunix.htm'
];
var list_down2 = [
'install.htm',
'introduction.htm',
'ad.htm',
'adfun.htm',
'library.htm',
'preprocessor.htm',
'example.htm',
'appendix.htm'
];
var list_down1 = [
'installunix.htm',
'installwindows.htm'
];
var list_down0 = [
'subversion.htm'
];
var list_current0 = [
'installunix.htm#Fedora',
'installunix.htm#RPM',
'installunix.htm#Download',
'installunix.htm#Download.Subversion',
'installunix.htm#Download.Web Link',
'installunix.htm#Download.Unix Tar Files',
'installunix.htm#Download.Tar File Extraction',
'installunix.htm#Configure',
'installunix.htm#Testing Return Status',
'installunix.htm#PrefixDir',
'installunix.htm#--with-Documentation',
'installunix.htm#--with-Introduction',
'installunix.htm#--with-Introduction.get_started',
'installunix.htm#--with-Introduction.exp_apx',
'installunix.htm#--with-Example',
'installunix.htm#--with-TestMore',
'installunix.htm#--with-Speed',
'installunix.htm#--with-Speed.cppad',
'installunix.htm#--with-Speed.double',
'installunix.htm#--with-Speed.profile',
'installunix.htm#--with-Speed.example',
'installunix.htm#--with-PrintFor',
'installunix.htm#--with-stdvector',
'installunix.htm#PostfixDir',
'installunix.htm#AdolcDir',
'installunix.htm#AdolcDir.Fix Adolc',
'installunix.htm#AdolcDir.Linux',
'installunix.htm#AdolcDir.Cygwin',
'installunix.htm#FadbadDir',
'installunix.htm#SacadoDir',
'installunix.htm#BoostDir',
'installunix.htm#IpoptDir',
'installunix.htm#CompilerFlags',
'installunix.htm#make',
'installunix.htm#make install'
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
