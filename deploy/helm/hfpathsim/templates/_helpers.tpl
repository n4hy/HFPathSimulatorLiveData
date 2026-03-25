{{/*
Expand the name of the chart.
*/}}
{{- define "hfpathsim.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "hfpathsim.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "hfpathsim.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "hfpathsim.labels" -}}
helm.sh/chart: {{ include "hfpathsim.chart" . }}
{{ include "hfpathsim.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "hfpathsim.selectorLabels" -}}
app.kubernetes.io/name: {{ include "hfpathsim.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
API selector labels
*/}}
{{- define "hfpathsim.api.selectorLabels" -}}
{{ include "hfpathsim.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
Web selector labels
*/}}
{{- define "hfpathsim.web.selectorLabels" -}}
{{ include "hfpathsim.selectorLabels" . }}
app.kubernetes.io/component: web
{{- end }}

{{/*
API service name
*/}}
{{- define "hfpathsim.api.serviceName" -}}
{{ include "hfpathsim.fullname" . }}-api
{{- end }}

{{/*
Web service name
*/}}
{{- define "hfpathsim.web.serviceName" -}}
{{ include "hfpathsim.fullname" . }}-web
{{- end }}
