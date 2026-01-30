
import os
import importlib.util
import appdirs

#Set up a function to load modules/scripts in the environment
def load_additional_modules(dirname, all_modules):
    #Loop over all py files in the directory:
    for f in os.listdir(dirname):
        #Check if they should be appended, and if so, append them to all
        if f != "__init__.py" and os.path.isfile(os.path.join(dirname, f)) and f.endswith(".py"):
            __all__.append(f[:-3])
        
    #Loop over all and add them as import
    for module_name in all_modules:
        try:
            #First try the normal import
            exec(f"from .{module_name} import *")
            
            # Also import the module itself so it can be referenced by name (required for utils.functionNamesFromDir)
            # This is a bit of a hack to ensure eval('module_name') works
            import importlib
            globals()[module_name] = importlib.import_module(f".{module_name}", package=__name__)
            
        except ModuleNotFoundError:
            #Otherwise import via importlib (from appdata folder)
            module_path = os.path.join(dirname, f"{module_name}.py")
            module_name_without_extension = os.path.splitext(module_name)[0]
            spec = importlib.util.spec_from_file_location(module_name_without_extension, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            globals()[module_name_without_extension] = module  # Update globals with the module itself

__all__ = []
#First load modules from this exact path
dirname = os.path.dirname(os.path.abspath(__file__))
load_additional_modules(dirname,__all__)
#Then add additional modules from the relevant appdirectory
app_data_dir = appdirs.user_data_dir()
last_component = os.path.basename(dirname)
appdata_folder = os.path.join(app_data_dir, 'UniBonn','Eve',last_component)
#Check if this folder exists, if not, create it:
if not os.path.isdir(appdata_folder):
    os.makedirs(appdata_folder)
#Load modules from the appdata folder
load_additional_modules(appdata_folder,__all__)
