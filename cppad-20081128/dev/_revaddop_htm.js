var list_across0 = [
'_contents.htm',
'_reference.htm',
'_index.htm',
'_search.htm',
'_external.htm'
];
var list_up0 = [
'cppad.htm',
'op.htm',
'revaddop.htm'
];
var list_down2 = [
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
var list_down1 = [
'forabsop.htm',
'revabsop.htm',
'foraddop.htm',
'revaddop.htm',
'foracosop.htm',
'revacosop.htm',
'forasinop.htm',
'revasinop.htm',
'foratanop.htm',
'revatanop.htm',
'fordivvvop.htm',
'revdivvvop.htm',
'forexpop.htm',
'revexpop.htm',
'forlogop.htm',
'revlogop.htm',
'formulvvop.htm',
'revmulvvop.htm',
'forsincos.htm',
'revsincos.htm',
'forsqrtop.htm',
'revsqrtop.htm',
'forsubvvop.htm',
'revsubvvop.htm'
];
var list_current0 = [
'revaddop.htm#Syntax',
'revaddop.htm#Description',
'revaddop.htm#pz',
'revaddop.htm#On Input',
'revaddop.htm#On Input.px',
'revaddop.htm#On Input.py',
'revaddop.htm#On Output',
'revaddop.htm#On Output.px',
'revaddop.htm#On Output.py'
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
