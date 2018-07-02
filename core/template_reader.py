# import tensorflow as tf

from NetBluePrint.core import builder


def create_block_operation(structure, name, argument_translation={}):
    def created_block(input, layer_id, construct_log, **kw):
        # print kw
        for arg in kw.keys():
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
        def_dict = dict(construct_log["default_dict"])
        def_dict.update(kw)
        ret, _ = builder.create_network(input, structure, name + "_" + str(layer_id), default_dict=def_dict,
                                        parent_log=construct_log)
        del construct_log["block_hierarchy"][-1]
        return ret

    return created_block


def block_repeater(input, layer_id, construct_log, structure, times, argument_translation={}, **kw):
    repeated_structure = []
    for i in range(times):
        repeated_structure += structure
    creator = create_block_operation(repeated_structure, "block_repeater", argument_translation)
    return creator(input, layer_id, construct_log, **kw)
