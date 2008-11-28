var list_across0 = [
'_contents.htm',
'_reference.htm',
'_index.htm',
'_search.htm',
'_external.htm'
];
var list_up0 = [
'cppad.htm',
'ad.htm',
'advalued.htm',
'arithmetic.htm',
'compute_assign.htm'
];
var list_down3 = [
'default.htm',
'ad_copy.htm',
'convert.htm',
'advalued.htm',
'boolvalued.htm',
'vecad.htm',
'base_require.htm'
];
var list_down2 = [
'arithmetic.htm',
'std_math_ad.htm',
'mathother.htm',
'condexp.htm',
'discrete.htm'
];
var list_down1 = [
'unaryplus.htm',
'unaryminus.htm',
'ad_binary.htm',
'compute_assign.htm'
];
var list_down0 = [
'addeq.cpp.htm',
'subeq.cpp.htm',
'muleq.cpp.htm',
'diveq.cpp.htm'
];
var list_current0 = [
'compute_assign.htm#Syntax',
'compute_assign.htm#Purpose',
'compute_assign.htm#Op',
'compute_assign.htm#Base',
'compute_assign.htm#x',
'compute_assign.htm#y',
'compute_assign.htm#Result',
'compute_assign.htm#Operation Sequence',
'compute_assign.htm#Example',
'compute_assign.htm#Derivative',
'compute_assign.htm#Derivative.Addition',
'compute_assign.htm#Derivative.Subtraction',
'compute_assign.htm#Derivative.Multiplication',
'compute_assign.htm#Derivative.Division'
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
function choose_down3(item)
{	var index          = item.selectedIndex;
	item.selectedIndex = 0;
	if(index > 0)
		document.location = list_down3[index-1];
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
