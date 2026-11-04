{% extends "python/module.rst" %}
{% block content %}
{% if obj.all %}
Public names
------------

The names that ``from {{ obj.id }} import ...`` provides, each linked to its
documentation.

.. list-table::
   :widths: 35 65

   {% for child in obj.children|sort(attribute="short_name", case_sensitive=True)
      if child.short_name in obj.all and not child.short_name.startswith("_") %}
   * - :py:obj:`{{ child.short_name }} <{{ child.id }}>`
     - {{ child.summary }}
   {% endfor %}

{% endif %}
{{ super() }}
{% endblock %}
