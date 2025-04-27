from jinja2 import Environment, FileSystemLoader
import os

def prompt_rendering(context, question):
    """
    Purpose: Render the prompt using Jinja2
    """
    prompts_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')
    env = Environment(loader=FileSystemLoader(prompts_dir))

    template = env.get_template('prompt_template.jinja')
    rendered_prompt = template.render(context=context, question=question)
    return rendered_prompt