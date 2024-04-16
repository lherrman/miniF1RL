
import json
class Config:
    @staticmethod
    def get(key):
        config = json.load(open('config.json', 'r'))
        if key in config:
            return config[key]

        for _, conf  in config.items():
            if key in conf:
                return conf[key]
            
        raise KeyError(f'Key {key} not found in config')
        
    @staticmethod
    def set(key, value):
        set_value = value
        # Check if value is a string that can be converted to a number
        if type(value) == str:
            try:
                set_value = float(value)
                if set_value.is_integer():
                    set_value = int(set_value)
            except:
                pass
        config = json.load(open('config.json', 'r'))
        for _, conf  in config.items():
            if key in conf:
                conf[key] = set_value
                json.dump(config, open('config.json', 'w'), indent=2)
                return
            
        raise KeyError(f'Key {key} not found in config')

    @staticmethod
    def add(group, key, value):
        config = json.load(open('config.json', 'r'))
        if group in config:
            if key in config[group]:
                raise KeyError(f'Key {key} already exists in config')
            config[group][key] = value
            json.dump(config, open('config.json', 'w'), indent=2)
            return
        raise KeyError(f'Group {group} not found in config')

    @staticmethod
    def get_groups():
        config = json.load(open('config.json', 'r'))
        return list(config.keys())
    
    @staticmethod
    def get_keys(group):
        config = json.load(open('config.json', 'r'))
        if group in config:
            return list(config[group].keys())
        raise KeyError(f'Group {group} not found in config')


from config import Config as cfg

if __name__ == '__main__':
    print(cfg.get('controll_frequency'))
    cfg.set('controll_frequency', 20)
    groups = cfg.get_groups()
    keys = cfg.get_keys(groups[0])
    print(keys)

    


