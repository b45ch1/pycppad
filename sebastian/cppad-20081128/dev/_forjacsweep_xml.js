var list_across0 = [
'_contents_xml.htm',
'_reference.xml',
'_index.xml',
'_search_xml.htm',
'_external.xml'
];
var list_up0 = [
'cppad.xml',
'forjacsweep.xml'
];
var list_down1 = [
'distribute.xml',
'newfeature.xml',
'define.xml',
'greaterthanzero.xml',
'greaterthanorzero.xml',
'lessthanzero.xml',
'lessthanorzero.xml',
'identicalpar.xml',
'identicalzero.xml',
'identicalone.xml',
'identicalequalpar.xml',
'opcode.xml',
'printop.xml',
'numind.xml',
'numvar.xml',
'tape_link.xml',
'recorder.xml',
'player.xml',
'adtape.xml',
'boolfunlink.xml',
'op.xml',
'forward_sweep.xml',
'reversesweep.xml',
'forjacsweep.xml',
'revjacsweep.xml'
];
var list_current0 = [
'forjacsweep.xml#Syntax',
'forjacsweep.xml#Rec',
'forjacsweep.xml#Description',
'forjacsweep.xml#numvar',
'forjacsweep.xml#npv',
'forjacsweep.xml#TaylorColDim',
'forjacsweep.xml#Taylor',
'forjacsweep.xml#On Input',
'forjacsweep.xml#On Input.Independent Variables and Operators',
'forjacsweep.xml#On Input.Other Variables and Operators',
'forjacsweep.xml#On Output',
'forjacsweep.xml#On Output.Independent Variables',
'forjacsweep.xml#On Output.Other Variables'
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
