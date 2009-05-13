var list_across0 = [
'_contents.htm',
'_reference.htm',
'_index.htm',
'_search.htm',
'_external.htm'
];
var list_up0 = [
'cppad.htm',
'forward_sweep.htm'
];
var list_down1 = [
'distribute.htm',
'newfeature.htm',
'define.htm',
'greaterthanzero.htm',
'greaterthanorzero.htm',
'lessthanzero.htm',
'lessthanorzero.htm',
'identicalpar.htm',
'identicalzero.htm',
'identicalone.htm',
'identicalequalpar.htm',
'opcode.htm',
'printop.htm',
'numind.htm',
'numvar.htm',
'tape_link.htm',
'recorder.htm',
'player.htm',
'adtape.htm',
'boolfunlink.htm',
'op.htm',
'forward_sweep.htm',
'reversesweep.htm',
'forjacsweep.htm',
'revjacsweep.htm'
];
var list_down0 = [
'forward0sweep.htm'
];
var list_current0 = [
'forward_sweep.htm#Syntax',
'forward_sweep.htm#Return Value',
'forward_sweep.htm#Rec',
'forward_sweep.htm#print',
'forward_sweep.htm#d',
'forward_sweep.htm#numvar',
'forward_sweep.htm#J',
'forward_sweep.htm#On Input',
'forward_sweep.htm#On Input.Independent Variables and Operators',
'forward_sweep.htm#On Input.Other Variables and Operators',
'forward_sweep.htm#On Output',
'forward_sweep.htm#On Output.Rec',
'forward_sweep.htm#On Output.Independent Variables',
'forward_sweep.htm#On Output.Other Variables',
'forward_sweep.htm#Contents'
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
