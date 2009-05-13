var list_across0 = [
'_contents_xml.htm',
'_reference.xml',
'_index.xml',
'_search_xml.htm',
'_external.xml'
];
var list_up0 = [
'cppad.xml',
'ad.xml',
'base_require.xml'
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
'default.xml',
'ad_copy.xml',
'convert.xml',
'advalued.xml',
'boolvalued.xml',
'vecad.xml',
'base_require.xml'
];
var list_down0 = [
'base_complex.hpp.xml',
'base_adolc.hpp.xml'
];
var list_current0 = [
'base_require.xml#Purpose',
'base_require.xml#Warning',
'base_require.xml#Numeric Type',
'base_require.xml#declare.hpp',
'base_require.xml#CondExp',
'base_require.xml#CondExp.Ordered Type',
'base_require.xml#CondExp.Not Ordered',
'base_require.xml#EqualOpSeq',
'base_require.xml#EqualOpSeq.Suggestion',
'base_require.xml#Identical',
'base_require.xml#Identical.Suggestion',
'base_require.xml#Integer',
'base_require.xml#Integer.Suggestion',
'base_require.xml#Ordered',
'base_require.xml#Ordered.Ordered Type',
'base_require.xml#Ordered.Not Ordered',
'base_require.xml#pow',
'base_require.xml#Standard Math Unary',
'base_require.xml#Example'
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
