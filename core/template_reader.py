# import tensorflow as tf

from NetBluePrint.core import builder
from .path_dict import PathDict


def create_block_operation(structure, name, argument_translation={}, default_parameters={}, scope_type="VAR"):
    def created_block(input, layer_id, construct_log, *args, **kw):
        # print kw
        construct_log["printer"].printResult("INFO", "Template "+name + " pre-evaluation arguments")
        if len(args)!=0:
            if "args" not in list(kw.keys()):
                kw["args"] = []
            kw["args"] = args + kw["args"]
        for def_arg in list(default_parameters.keys()):
            if def_arg not in list(kw.keys()):
                kw[def_arg] = default_parameters[def_arg]
        for arg in list(kw.keys()):
            if arg in argument_translation:
                if type(argument_translation[arg]) is list:
                    for at in argument_translation[arg]:
                        kw[at] = kw[arg]
                else:
                    kw[argument_translation[arg]] = kw[arg]
                del kw[arg]
        if "block_hierarchy" not in construct_log:
            construct_log["block_hierarchy"] = []
        if name in construct_log["block_hierarchy"]:
            construct_log["printer"].printError(name + " appears multiple times in block_hierarchy.")
            construct_log["printer"].printError(
                "Can be caused by invalid block declaration or improper cleaning of block_hierarchy.")
            raise Exception("block_hierarchy error")
        construct_log["block_hierarchy"].append(name)
        def_dict = PathDict(construct_log["default_dict"])
        def_dict.update(kw)
        construct_log["printer"].printResult("INFO", "Template "+name + " saving template in logger")
        construct_log["logger"].register_opp({"structure" : structure,
                                                             "argument_translation" : argument_translation,
                                                             "default_parameters" : default_parameters},name, "template")
        construct_log["printer"].printResult("INFO", "Template "+name + " starting sub-workflow creation")
        ret, _ = builder.create_workflow(input, structure, name + "_" + str(layer_id), default_dict=def_dict,
                                         parent_log=construct_log, scope_type=scope_type)
        del construct_log["block_hierarchy"][-1]
        return ret

    return created_block