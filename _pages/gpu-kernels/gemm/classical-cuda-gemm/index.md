---
layout: page
title: Part I — Classical CUDA GEMM
permalink: /gpu-kernels/gemm/classical-cuda-gemm/
nav: false
series: classical-cuda-gemm
---

[← GEMM]({{ '/gpu-kernels/gemm/' | relative_url }})

{% assign series_posts = site.posts | where: 'series', page.series | sort: 'chapter' %}
{% if series_posts.size > 0 %}
  <ol>
    {% for post in series_posts %}
      <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ol>
{% endif %}
