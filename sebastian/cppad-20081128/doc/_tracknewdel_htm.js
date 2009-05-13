var list_across0 = [
'_contents.htm',
'_reference.htm',
'_index.htm',
'_search.htm',
'_external.htm'
];
var list_up0 = [
'cppad.htm',
'library.htm',
'tracknewdel.htm'
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
'errorhandler.htm',
'nearequal.htm',
'speed_test.htm',
'speedtest.htm',
'numerictype.htm',
'checknumerictype.htm',
'simplevector.htm',
'checksimplevector.htm',
'nan.htm',
'pow_int.htm',
'poly.htm',
'ludetandsolve.htm',
'rombergone.htm',
'rombergmul.htm',
'runge45.htm',
'rosen34.htm',
'odeerrcontrol.htm',
'odegear.htm',
'odegearcontrol.htm',
'benderquad.htm',
'luratio.htm',
'std_math_unary.htm',
'cppad_vector.htm',
'tracknewdel.htm'
];
var list_down0 = [
'tracknewdel.cpp.htm'
];
var list_current0 = [
'tracknewdel.htm#Syntax',
'tracknewdel.htm#Purpose',
'tracknewdel.htm#Include',
'tracknewdel.htm#file',
'tracknewdel.htm#line',
'tracknewdel.htm#oldptr',
'tracknewdel.htm#oldptr.OpenMP',
'tracknewdel.htm#newlen',
'tracknewdel.htm#head newptr',
'tracknewdel.htm#ncopy',
'tracknewdel.htm#TrackNewVec',
'tracknewdel.htm#TrackNewVec.Macro',
'tracknewdel.htm#TrackNewVec.Deprecated',
'tracknewdel.htm#TrackDelVec',
'tracknewdel.htm#TrackDelVec.Macro',
'tracknewdel.htm#TrackDelVec.Deprecated',
'tracknewdel.htm#TrackExtend',
'tracknewdel.htm#TrackExtend.Macro',
'tracknewdel.htm#TrackExtend.Deprecated',
'tracknewdel.htm#TrackCount',
'tracknewdel.htm#TrackCount.Macro',
'tracknewdel.htm#TrackCount.Deprecated',
'tracknewdel.htm#TrackCount.OpenMP',
'tracknewdel.htm#Example'
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
