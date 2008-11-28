var list_across0 = [
'_contents.htm',
'_reference.htm',
'_index.htm',
'_search.htm',
'_external.htm'
];
var list_up0 = [
'cppad.htm',
'adfun.htm',
'funconstruct.htm'
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
'independent.htm',
'funconstruct.htm',
'dependent.htm',
'seqproperty.htm',
'funeval.htm',
'drivers.htm',
'funcheck.htm',
'omp_max_thread.htm',
'fundeprecated.htm'
];
var list_current0 = [
'funconstruct.htm#Syntax',
'funconstruct.htm#Purpose',
'funconstruct.htm#x',
'funconstruct.htm#y',
'funconstruct.htm#Default Constructor',
'funconstruct.htm#Sequence Constructor',
'funconstruct.htm#OpenMP',
'funconstruct.htm#Example',
'funconstruct.htm#Example.Sequence Constructor',
'funconstruct.htm#Example.Default Constructor'
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
