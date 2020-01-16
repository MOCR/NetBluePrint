
class PathDict:
    def __init__(self, init_dict={}):
        self.internal_dict = {}
        self.internal_dict.update(init_dict)

    def __getitem__(self, key):
        if type(key) != list:
            key = key.split("/")
        local_key = key[0]
        child_key = key[1:]

        local_node = self.internal_dict[local_key]
        if len(child_key) > 0:
            return local_node[child_key]
        else:
            return local_node

    def __setitem__(self, key, value):
        if type(key) is not list:
            key = key.split("/")
        local_key = key[0]
        child_key = key[1:]

        if len(child_key) > 0:
            if local_key not in self.internal_dict:
                self.internal_dict[local_key]=PathDict()
            self.internal_dict[local_key][child_key] = value
        else:
            self.internal_dict[local_key] = value


    def __delitem__(self, key):
        if type(key) != list:
            key = key.split("/")
        local_key = key[0]
        child_key = key[1:]

        if len(child_key) > 0:
            del self.internal_dict[local_key][child_key]
        else:
            del self.internal_dict[local_key]

    def keys(self, leafs=False, recursive=False, node_path=None):
        local_keys = list(self.internal_dict.keys())

        keys = []
        for k in local_keys:
            if node_path is not None:
                full_key = node_path+"/"+k
            else:
                full_key = k
            if isinstance(self.internal_dict[k], PathDict):
                if recursive:
                    keys += self.internal_dict[k].keys(leafs=leafs, recursive=recursive, node_path=full_key)
                if not leafs:
                    keys.append(full_key)
            else:
                keys.append(full_key)
        return keys

    def values(self):
        return self.internal_dict.values()