from abc import ABC, abstractmethod
import yaml


class ConfigHolder(ABC):

    @abstractmethod
    def get_as_dict(self):
        raise NotImplementedError

    def to_yaml(self, path):
        params = self.get_as_dict()
        with open(path, 'w') as outfile:
            yaml.dump(params, outfile)
