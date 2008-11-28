var list_across0 = [
'_contents_xml.htm',
'_reference.xml',
'_index.xml',
'_search_xml.htm',
'_external.xml'
];
var list_up0 = [
'cppad.xml',
'library.xml',
'tracknewdel.xml'
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
'errorhandler.xml',
'nearequal.xml',
'speed_test.xml',
'speedtest.xml',
'numerictype.xml',
'checknumerictype.xml',
'simplevector.xml',
'checksimplevector.xml',
'nan.xml',
'pow_int.xml',
'poly.xml',
'ludetandsolve.xml',
'rombergone.xml',
'rombergmul.xml',
'runge45.xml',
'rosen34.xml',
'odeerrcontrol.xml',
'odegear.xml',
'odegearcontrol.xml',
'benderquad.xml',
'luratio.xml',
'std_math_unary.xml',
'cppad_vector.xml',
'tracknewdel.xml'
];
var list_down0 = [
'tracknewdel.cpp.xml'
];
var list_current0 = [
'tracknewdel.xml#Syntax',
'tracknewdel.xml#Purpose',
'tracknewdel.xml#Include',
'tracknewdel.xml#file',
'tracknewdel.xml#line',
'tracknewdel.xml#oldptr',
'tracknewdel.xml#oldptr.OpenMP',
'tracknewdel.xml#newlen',
'tracknewdel.xml#head newptr',
'tracknewdel.xml#ncopy',
'tracknewdel.xml#TrackNewVec',
'tracknewdel.xml#TrackNewVec.Macro',
'tracknewdel.xml#TrackNewVec.Deprecated',
'tracknewdel.xml#TrackDelVec',
'tracknewdel.xml#TrackDelVec.Macro',
'tracknewdel.xml#TrackDelVec.Deprecated',
'tracknewdel.xml#TrackExtend',
'tracknewdel.xml#TrackExtend.Macro',
'tracknewdel.xml#TrackExtend.Deprecated',
'tracknewdel.xml#TrackCount',
'tracknewdel.xml#TrackCount.Macro',
'tracknewdel.xml#TrackCount.Deprecated',
'tracknewdel.xml#TrackCount.OpenMP',
'tracknewdel.xml#Example'
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
