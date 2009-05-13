var list_across0 = [
'_contents.htm',
'_reference.htm',
'_index.htm',
'_search.htm',
'_external.htm'
];
var list_up0 = [
'cppad.htm',
'player.htm'
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
var list_current0 = [
'player.htm#Syntax',
'player.htm#Default Constructors',
'player.htm#Assignment Operator',
'player.htm#Erase',
'player.htm#GetOp',
'player.htm#GetInd',
'player.htm#GetPar',
'player.htm#GetVecInd',
'player.htm#NumOp',
'player.htm#NumInd',
'player.htm#NumPar',
'player.htm#NumVecInd',
'player.htm#ReplaceInd',
'player.htm#TotNumVar',
'player.htm#Memory'
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
